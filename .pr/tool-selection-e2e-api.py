import json
import os
import sys
import time
from pathlib import Path

import httpx

S = Path(os.environ["E2E_PERSISTENCE_DIR"])
KEY = (S / "api-key.txt").read_text().strip()
BASE = "http://127.0.0.1:18300"
MOCK = "http://127.0.0.1:18399"
EXPECT_BROWSER = os.environ.get("E2E_EXPECT_BROWSER", "1") == "1"
c = httpx.Client(base_url=BASE, headers={"X-Session-API-Key": KEY}, timeout=60, trust_env=False)
m = httpx.Client(base_url=MOCK, timeout=30, trust_env=False)
results = []


def check(name, ok, evidence):
    results.append((name, bool(ok), evidence))
    print(("PASS " if ok else "FAIL ") + name + " :: " + evidence, flush=True)


def names(tools):
    return None if tools is None else [t["name"] for t in tools]


def has_browser(tools):
    return any(name.startswith("browser") for name in tools or [])


# a. catalog
cat = c.get("/api/tools/catalog").json()["tools"]
default = sorted(t["name"] for t in cat if t["in_default_set"])
check("a.catalog", "switch_llm" in default and "terminal" in default,
      f"{len(cat)} entries; default={default}")
browser_usable = next(t["usable"] for t in cat if t["name"] == "browser_tool_set")
check("a.catalog-browser-usable", browser_usable is EXPECT_BROWSER,
      f"browser_tool_set usable={browser_usable}")

# LLM profile
r = c.post("/api/profiles/e2e-mock", json={"llm": {"model": "openai/mock-test-model", "base_url": f"{MOCK}/v1", "api_key": "mock"}, "include_secrets": True})
assert r.status_code < 300, r.text


def prof(name, tools, **extra):
    body = {"name": name, "agent_kind": "openhands", "llm_profile_ref": "e2e-mock",
            "mcp_server_refs": [], "secret_refs": [], "condenser": {"enabled": False}, **extra}
    if tools is not ...:
        body["tools"] = tools
    return c.post(f"/api/agent-profiles/{name}", json=body)


# b. profiles
r1 = prof("e2e-unset", ...)
g1 = c.get("/api/agent-profiles/e2e-unset").json()["profile"]
check("b.create-unset", r1.status_code == 201 and g1.get("tools") is None
      and "enable_sub_agents" not in g1 and "enable_switch_llm_tool" not in g1,
      f"{r1.status_code}; tools={g1.get('tools')}; keys-with-enable={[k for k in g1 if k.startswith('enable_')]}")
r2 = prof("e2e-explicit", [{"name": "terminal"}, {"name": "glob"}])
g2 = c.get("/api/agent-profiles/e2e-explicit").json()["profile"]
check("b.create-explicit", r2.status_code == 201 and names(g2["tools"]) == ["terminal", "glob"],
      f"{r2.status_code}; tools={names(g2['tools'])}")
# save-as-new: GET unset-tools profile, set explicit tools, POST under a new name
copy = {k: v for k, v in g1.items() if k not in ("id", "revision")}
copy["tools"] = [{"name": "terminal"}, {"name": "glob"}]
copy["name"] = "e2e-copy"
r3 = c.post("/api/agent-profiles/e2e-copy", json=copy)
g3 = c.get("/api/agent-profiles/e2e-copy").json()["profile"]
check("b.save-as-new", r3.status_code == 201 and names(g3["tools"]) == ["terminal", "glob"],
      f"{r3.status_code}; tools={names(g3['tools'])}")
# GET explicit then re-POST as new name verbatim
copy2 = {k: v for k, v in g2.items() if k not in ("id", "revision")}
r3b = c.post("/api/agent-profiles/e2e-copy2", json={**copy2, "name": "e2e-copy2"})
g3b = c.get("/api/agent-profiles/e2e-copy2").json()["profile"]
check("b.roundtrip-copy", r3b.status_code == 201 and names(g3b["tools"]) == ["terminal", "glob"],
      f"{r3b.status_code}; tools={names(g3b['tools'])}")
r4 = prof("e2e-flag", ..., enable_sub_agents=True)
g4 = c.get("/api/agent-profiles/e2e-flag").json()["profile"]
check("b.flag-folded", r4.status_code == 201 and "task_tool_set" in names(g4["tools"])
      and "enable_sub_agents" not in g4, f"{r4.status_code}; tools={names(g4['tools'])}")

# d. materialize
r5 = c.post("/api/agent-profiles/e2e-bogus/materialize", json={"profile": {
    "name": "e2e-bogus", "agent_kind": "openhands", "llm_profile_ref": "e2e-mock",
    "tools": [{"name": "terminal"}, {"name": "definitely_not_a_tool"}, {"name": "browser_tool_set"}]}})
d = r5.json()
check("d.materialize-unregistered", r5.status_code == 200 and d.get("valid") is False
      and "definitely_not_a_tool" in (d.get("unusable_tools") or []),
      f"{r5.status_code}; valid={d.get('valid')}; unusable_tools={d.get('unusable_tools')}; errors={d.get('errors')}")
r5b = c.post("/api/agent-profiles/e2e-ok/materialize", json={"profile": {
    "name": "e2e-ok", "agent_kind": "openhands", "llm_profile_ref": "e2e-mock",
    "tools": [{"name": "terminal"}, {"name": "glob"}]}})
d = r5b.json()
check("d.materialize-valid", r5b.status_code == 200 and d.get("valid") is True and not d.get("unusable_tools"),
      f"{r5b.status_code}; valid={d.get('valid')}; unusable_tools={d.get('unusable_tools')}")
rs = d.get("resolved_settings") or {}
check("d.resolved-no-flags", "enable_sub_agents" not in rs and "enable_switch_llm_tool" not in rs,
      f"resolved_settings enable_* keys={[k for k in rs if k.startswith('enable_')]}")

# e. settings
s0 = c.get("/api/settings").json()
ag = s0.get("agent_settings", {})
check("e.settings-no-flags", "enable_sub_agents" not in ag and "enable_switch_llm_tool" not in ag,
      f"agent_settings enable_* keys={[k for k in ag if k.startswith('enable_')]}; tools={names(ag.get('tools'))}")
sch = json.dumps(c.get("/api/settings/agent-schema").json())
check("e.schema-no-flags", "enable_sub_agents" not in sch and "enable_switch_llm_tool" not in sch,
      f"agent-schema mentions flags={'enable_sub_agents' in sch or 'enable_switch_llm_tool' in sch}")
r6 = c.patch("/api/settings", json={"agent_settings_diff": {"enable_sub_agents": True}})
s1 = c.get("/api/settings").json()["agent_settings"]
check("e.patch-flag-folds", r6.status_code == 200 and "task_tool_set" in (names(s1.get("tools")) or [])
      and "enable_sub_agents" not in s1,
      f"{r6.status_code}; tools={names(s1.get('tools'))}")
# reset to default tools
r7 = c.patch("/api/settings", json={"agent_settings_diff": {"tools": None}})
s2 = c.get("/api/settings").json()["agent_settings"]
check("e.reset-tools-null", r7.status_code == 200 and s2.get("tools") is None, f"{r7.status_code}; tools={s2.get('tools')}")


# f. conversations
def run_conv(label, **launch):
    m.post("/admin/reset")
    m.post("/admin/trajectory/register", json={"name": "fin", "turns": [
        {"tool_call": {"name": "finish", "arguments": {"message": "OK"}}} for _ in range(3)]})
    m.post("/admin/trajectory/activate", json={"name": "fin"})
    ws = S / "e2e-workspace" / label
    ws.mkdir(parents=True, exist_ok=True)
    body = {"workspace": {"working_dir": str(ws)}, "max_iterations": 2, "title": label,
            "initial_message": {"role": "user", "content": [{"type": "text", "text": "finish"}]}, **launch}
    r = c.post("/api/conversations", json=body)
    if r.status_code >= 300:
        return None, f"create {r.status_code}: {r.text[:300]}"
    cid = r.json()["id"]
    st = None
    for _ in range(90):
        st = c.get(f"/api/conversations/{cid}").json().get("execution_status")
        if st in ("finished", "error", "stuck"):
            break
        time.sleep(0.5)
    reqs = m.get("/admin/requests").json()
    calls = reqs if isinstance(reqs, list) else reqs.get("requests", [])
    tools = [t["function"]["name"] for t in calls[-1].get("tools", [])] if calls else []
    return tools, f"status={st}; llm tools={tools}"


def llm_settings():
    return {"schema_version": 7, "agent_kind": "openhands",
            "llm": {"model": "openai/mock-test-model", "base_url": f"{MOCK}/v1", "api_key": "mock"}}


ids = {n: c.get(f"/api/agent-profiles/{n}").json()["profile"]["id"] for n in ("e2e-unset", "e2e-explicit")}
prof("e2e-subagents", [{"name": "terminal"}, {"name": "task_tool_set"}])
ids["e2e-subagents"] = c.get("/api/agent-profiles/e2e-subagents").json()["profile"]["id"]

t, ev = run_conv("default-settings", agent_settings=llm_settings())
check("f.default-settings", t is not None and {"terminal", "file_editor", "task_tracker", "switch_llm"} <= set(t or [])
      and len(set(t or [])) == len(t or []) and has_browser(t) is EXPECT_BROWSER, ev)
t, ev = run_conv("explicit-settings", agent_settings={**llm_settings(), "tools": [{"name": "terminal"}, {"name": "browser_tool_set"}]})
check("f.explicit-settings", t is not None and "terminal" in t and "switch_llm" not in t and has_browser(t) is EXPECT_BROWSER, ev)
t, ev = run_conv("profile-unset", agent_profile_id=ids["e2e-unset"])
check("f.profile-unset", t is not None and {"terminal", "switch_llm"} <= set(t or []) and has_browser(t) is EXPECT_BROWSER, ev)
t, ev = run_conv("profile-explicit", agent_profile_id=ids["e2e-explicit"])
opt = set(t or []) - {"finish", "think", "invoke_skill"}
check("f.profile-explicit", t is not None and opt == {"terminal", "glob"}, ev)
t, ev = run_conv("profile-subagents", agent_profile_id=ids["e2e-subagents"])
check("f.profile-subagents", t is not None and "task" in (t or []) or any("task" == x or x.startswith("task") and x != "task_tracker" for x in (t or [])), ev)


print("FAILED:", [r[0] for r in results if not r[1]])
sys.exit(0)
