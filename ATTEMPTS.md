# Critic Feature: Staging Debugging & Deployment Attempts

## 目标

让 PR #14133 (feat: Add critic result visualization to GUI frontend) 在 staging 上正常工作，使 critic_result 不再为 None。

PR: https://github.com/OpenHands/OpenHands/pull/14133

---

## 背景

### Critic 是什么
Critic 是一个自建的免费模型（`critic-qwen3-4b`），在 agent 完成任务或发消息时对结果做评估（打分、分类）。PR #14133 在前端加了 critic 结果的可视化展示。

### 架构
```
用户 → App Server → SDK Agent → Critic (APIBasedCritic)
                                    ↓
                          POST {server_url}/classify
                          Authorization: Bearer {api_key}
```

- **默认 critic endpoint**（Modal）: `https://all-hands-ai--critic-qwen3-4b-serve.modal.run`
- **Staging 覆盖**（走 LiteLLM proxy）: `https://llm-proxy.staging.all-hands.dev/vllm`
- **api_key**: 继承自 `self.llm.api_key`（用户的 LiteLLM proxy key）

### 关键代码位置
| 文件 | 作用 |
|------|------|
| `openhands-sdk/.../critic/impl/api/client.py` | CriticClient，发 `/classify` 请求 |
| `openhands-sdk/.../settings/model.py:747` | `build_critic()` — 用 `self.llm.api_key` 作为 critic key |
| `openhands/app_server/config.py` | `get_critic_server_url()` — 从 `OPENHANDS_PROVIDER_BASE_URL` 派生 |
| `openhands/app_server/.../live_status_app_conversation_service.py:1417` | 部署级 critic URL/model 注入 |
| `enterprise/storage/lite_llm_manager.py` | LiteLLM key 创建 & budget 管理 |

---

## 根因分析

### 发现过程

1. 在 staging 上测试 conversation `465beaa63a7d49159194b6f1d718cb26`，发现 `critic_result` 全是 `None`
2. 确认 critic 配置正确：`critic_enabled=true`, `critic_mode=finish_and_message`, `critic_server_url=https://llm-proxy.staging.all-hands.dev/vllm`, `critic_model_name=critic-qwen3-4b`
3. 直接 curl 测试 critic endpoint：

```bash
curl "https://llm-proxy.staging.all-hands.dev/vllm/classify" \
  -H "Authorization: Bearer sk-R4y5IYMv9Bo3Hf3g9usxzQ" \
  -d '{"model": "critic-qwen3-4b", "input": "test"}'
```

返回：
```json
{
  "error": {
    "message": "Budget has been exceeded! User=c1ec668f-5a9f-4075-9ea9-fb35e4f82f09 in Team=c1ec668f-5a9f-4075-9ea9-fb35e4f82f09 Current cost: 0.0, Max budget: 0.0",
    "type": "budget_exceeded"
  }
}
```

4. 主 LLM 同样被拦截（同一个 key，同一个 proxy）：
```bash
curl "https://llm-proxy.staging.all-hands.dev/chat/completions" \
  -H "Authorization: Bearer sk-R4y5IYMv9Bo3Hf3g9usxzQ" \
  -d '{"model": "litellm_proxy/claude-sonnet-4-20250514", "messages": [{"role":"user","content":"say hi"}]}'
# → 同样返回 budget_exceeded
```

### 根因

Staging 部署设置了 `ENABLE_BILLING=true`。`lite_llm_manager.py` 中创建新用户的逻辑：

```python
ENABLE_BILLING = os.environ.get('ENABLE_BILLING', 'false').lower() == 'true'

def _get_default_initial_budget():
    if not ENABLE_BILLING:
        return None  # billing 关 → budget=null（无限）
    budget = float(os.environ.get('DEFAULT_INITIAL_BUDGET', 0.0))  # billing 开 → 默认 $0
    return budget
```

- Staging `ENABLE_BILLING=true` → `DEFAULT_INITIAL_BUDGET=0.0` → 新用户 budget = $0
- LiteLLM 的 budget 检查：`if current_cost >= max_budget` → `0 >= 0` → 永远拒绝
- Critic 调用 `/vllm/classify` 走 LiteLLM proxy，被 budget 拦截
- 错误在 SDK 的 `critic_mixin.py` 中被静默吞掉（`except Exception: return None`）

### 为什么 critic 应该免费
Critic 模型是自建的（`critic-qwen3-4b`），部署在 Modal 或 vLLM 上，不消耗外部 LLM 额度。但因为 staging 把 critic 路由到 LiteLLM proxy（`/vllm` path），proxy 对所有请求执行 budget 检查，不区分模型是否免费。

---

## 已做的改动

### Commit 1: `73b25de` — 前端 critic 可视化（PR 原始内容）
- 前端加 critic result 组件、类型定义、事件渲染

### Commit 2: `1b7900e` — 加 CRITIC_API_KEY 环境变量
**文件**: `config.py`, `live_status_app_conversation_service.py`

目的：允许部署提供一个独立的 service-level API key 给 critic，不受用户 budget 限制。

```python
# config.py
def get_critic_api_key() -> str | None:
    return os.getenv('CRITIC_API_KEY') or None

# AppServerConfig
critic_api_key: str | None = Field(default_factory=get_critic_api_key)

# live_status_app_conversation_service.py — agent 创建后注入
if agent.critic is not None and self.critic_api_key:
    agent.critic.api_key = SecretStr(self.critic_api_key)
```

**状态**: 代码已推送，但 staging 部署未设置 `CRITIC_API_KEY` 值，所以暂时不生效。

### Commit 3: `b84afaf` — 改默认 budget 为 $10
**文件**: `enterprise/storage/lite_llm_manager.py`

```python
# 改前
budget = float(os.environ.get('DEFAULT_INITIAL_BUDGET', 0.0))
# 改后
budget = float(os.environ.get('DEFAULT_INITIAL_BUDGET', 10.0))
```

**状态**: 代码已推送。只对**新创建的用户**生效。已有用户（如测试用户 `c1ec668f-...`）的 budget 不变。

---

## 部署记录

| 部署 | Deploy PR | 状态 |
|------|-----------|------|
| `ohpr-14133-672` | deploy#3961 | 旧部署，不含 budget 修复 |
| `ohpr-14133-308` | deploy#3968 | 新部署，含所有 3 个 commit |

部署通过 `deploy` 仓库的 `Create OpenHands preview PR` workflow 触发：
```bash
gh workflow run "Create OpenHands preview PR" --repo OpenHands/deploy -f prNumber=14133
```

---

## 当前阻塞点

### 测试用户 budget = $0，需要手动修复

已有用户 `c1ec668f-5a9f-4075-9ea9-fb35e4f82f09` 的 team budget 仍然是 $0。需要用 LiteLLM admin key 调用：

```bash
curl -X POST "https://llm-proxy.staging.all-hands.dev/team/update" \
  -H "x-goog-api-key: <LITELLM_ADMIN_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"team_id": "c1ec668f-5a9f-4075-9ea9-fb35e4f82f09", "max_budget": 10.0}'
```

**问题**: LiteLLM admin key 在 deploy 仓库的 SOPS 加密文件中（`data_platform/envs/staging/secrets/litellm.yaml`），无法直接读取。

### 替代方案
1. **找有 staging infra 权限的人** 执行上面的 curl 命令
2. **删除并重建测试用户** — 重建时会用新的 $10 默认 budget
3. **注册新测试账号** — 会自动获得 $10 budget
4. **设置 CRITIC_API_KEY** — 在 staging 部署配置中加一个不受 budget 限制的 service key

---

## 长期方案建议

1. **Critic 应该完全绕过 budget** — 它是免费的自建模型，不该受 billing 限制
   - 已实现 `CRITIC_API_KEY` 机制（commit 2），部署时设置即可
   - 或者改 SDK 的 `build_critic()` 不继承用户的 LLM key
   
2. **DEFAULT_INITIAL_BUDGET 不应该是 $0** — 用户注册后什么都做不了
   - 已改为 $10（commit 3）
   - 也可以通过 staging 部署配置设 `DEFAULT_INITIAL_BUDGET` 环境变量

3. **Critic 错误不该被静默吞掉** — SDK 的 `critic_mixin.py` 捕获所有异常返回 None，调试困难
   - 建议至少在 event 中记录 critic 失败的原因
