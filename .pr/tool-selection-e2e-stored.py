import copy
import json
import os
from pathlib import Path

import httpx

S = Path(os.environ["E2E_PERSISTENCE_DIR"])
KEY = (S / "api-key.txt").read_text().strip()
c = httpx.Client(base_url="http://127.0.0.1:18300", headers={"X-Session-API-Key": KEY}, timeout=60, trust_env=False)
results = []


def check(name, ok, evidence):
    results.append((name, bool(ok), evidence))
    print(("PASS " if ok else "FAIL ") + name + " :: " + evidence, flush=True)


def names(tools):
    return None if tools is None else [t["name"] for t in tools]


base = json.loads((S / "agent-profiles/e2e-unset.json").read_text())
for key in ("id",):
    base.pop(key)
cases = {
    "e2e-v2-sub": ({"tools": None, "enable_sub_agents": True, "enable_switch_llm_tool": True},
                   lambda t: t is not None and "task_tool_set" in t and "switch_llm" in t and "terminal" in t),
    "e2e-v2-noswitch": ({"tools": [{"name": "terminal"}, {"name": "glob"}], "enable_sub_agents": False, "enable_switch_llm_tool": False},
                        lambda t: t == ["terminal", "glob"]),
    "e2e-v2-strflag": ({"tools": [{"name": "terminal"}], "enable_sub_agents": "false", "enable_switch_llm_tool": "true"},
                       lambda t: t == ["terminal", "switch_llm"]),
}
for name, (fields, ok) in cases.items():
    p = {**copy.deepcopy(base), "schema_version": 2, "name": name, **fields}
    (S / f"agent-profiles/{name}.json").write_text(json.dumps(p))
    r = c.get(f"/api/agent-profiles/{name}")
    got = r.json().get("profile", {}) if r.status_code == 200 else {}
    t = names(got.get("tools"))
    check(f"c.load-{name}", r.status_code == 200 and ok(t) and not any(k.startswith("enable_s") for k in got),
          f"{r.status_code}; tools={t}; flag keys={[k for k in got if k in ('enable_sub_agents','enable_switch_llm_tool')]}")
    body = {k: v for k, v in got.items() if k not in ("id", "revision")}
    r2 = c.post(f"/api/agent-profiles/{name}", json=body)
    disk = json.loads((S / f"agent-profiles/{name}.json").read_text())
    check(f"c.resave-{name}", r2.status_code == 201 and disk["schema_version"] == 3
          and "enable_sub_agents" not in disk and "enable_switch_llm_tool" not in disk and names(disk["tools"]) == t,
          f"{r2.status_code}; disk schema={disk['schema_version']}; disk tools={names(disk['tools'])}; flags on disk={[k for k in disk if k in ('enable_sub_agents','enable_switch_llm_tool')]}")

# v6 settings.json
sp = S / "settings.json"
orig = sp.read_text()
d = json.loads(orig)
a = d["agent_settings"]
a["schema_version"] = 6
a["tools"] = None
a["enable_sub_agents"] = True
a["enable_switch_llm_tool"] = False
sp.write_text(json.dumps(d))
try:
    g = c.get("/api/settings").json()["agent_settings"]
    t = names(g.get("tools"))
    check("e.v6-settings-load", t is not None and "task_tool_set" in t and "switch_llm" not in t and "enable_sub_agents" not in g,
          f"tools={t}; flag keys={[k for k in g if k in ('enable_sub_agents','enable_switch_llm_tool')]}")
    r = c.patch("/api/settings", json={"agent_settings_diff": {"tool_concurrency_limit": 1}})
    disk = json.loads(sp.read_text())["agent_settings"]
    check("e.v6-settings-resave", r.status_code == 200 and disk["schema_version"] == 7
          and "enable_sub_agents" not in disk and "enable_switch_llm_tool" not in disk and names(disk["tools"]) == t,
          f"{r.status_code}; disk schema={disk['schema_version']}; disk tools={names(disk['tools'])}")
finally:
    sp.write_text(orig)
g = c.get("/api/settings").json()["agent_settings"]
print("restored settings tools:", g.get("tools"))

print("FAILED:", [r[0] for r in results if not r[1]])
