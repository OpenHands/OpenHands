# Recording and Replaying a Session with OrcaReplay

[OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay) records an OpenHands session from outside the process and can then run it again against the recording — same conversation, same tool calls, no model called the second time. Nothing is installed into OpenHands: no plugin, no callback handler, no changes to your agent code.

> **Status:** verified against the **OpenHands SDK** (`openhands-ai` 1.11.0) used as a library. The packaged product running in its container runtime is **not** covered — see [Limits](#limits).

## Why you might want it

An agent session is expensive to reproduce. Re-running it costs tokens, hits the network, and the model may not make the same choices twice — so "what did it actually do, and why" is usually answered from logs rather than from the run itself.

OrcaReplay keeps the verbatim request and response bytes, so the recording can be served back:

- **Re-run a session for free.** No provider is called, no tokens are spent.
- **See the whole loop on one timeline.** Model calls with token counts, every tool call with its arguments, and a git tree hash per turn.
- **Local.** Traces are files under `.orca/` in your project. Nothing is uploaded.

## How the capture works

OpenHands' `LLM` wraps LiteLLM, which reads `OPENAI_API_BASE`. OrcaReplay sets that variable **for the child process it launches, and only for that process**, pointing it at a local proxy. Your agent code is unchanged and LiteLLM behaves exactly as it always does.

Pass the origin through the environment rather than `LLM(base_url=…)` in code — the environment is the route OrcaReplay sets up.

## Record a session

```bash
npm i -g orcareplay          # Node 20+
orca record generic-openai -- python your_agent.py
```

Using an ordinary SDK script:

```python
from openhands.sdk import LLM, Agent, Conversation
from openhands.tools.preset.default import get_default_tools

llm = LLM(model="openai/gpt-5.5", base_url=os.environ["OPENAI_API_BASE"], api_key=...)
agent = Agent(llm=llm, tools=get_default_tools(enable_browser=False))
convo = Conversation(agent=agent, workspace=os.getcwd())
convo.send_message("Read calc.py, fix the bug so add(2, 3) returns 5.")
convo.run()
```

```
info recorded run=run_d7670dd74f38 events=24 blobs=8 exit=0
```

## Read it back

```console
$ orca show last
run_d7670dd74f38  24 events  exit 0

SEQ  KIND   WHAT           DETAIL
2    MODEL  gpt-5.5        1 messages
3    MODEL  gpt-5.5        stop: tool_use · 7,047 in · 179 out
4    TOOL   terminal       {"command":"Get-ChildItem -Recurse -Filter calc.py …","security_risk":"LOW"}
9    TOOL   file_editor    {"command":"view","path":"…/calc.py"}
14   TOOL   file_editor    {"command":"str_replace","path":"…/calc.py"}
15   SNAP   tree af281cee…  1 changed
16   FILE   calc.py        modified +2 −2
```

That session started with a failing test and ended with a passing one; the diff it made is in the trace.

## Run it again against the recording

```console
$ orca replay last
info replaying exchanges=4 egress=blocked
info replay.done reused=4/4 divergences=4 unmatched=0 exit=0
```

The agent runs again, is served the recorded responses, and reaches the same result without any model being called.

The four `minor` divergences are honest output rather than noise: OpenHands puts **absolute paths and a session id** in the prompt, and both change between runs, so each request differs from the recording by 36–44 characters. Every exchange still matched — `reused=4/4` — and anything that drifts is named rather than waved through.

## Limits

Three, stated up front rather than discovered later.

**Shell exit codes are not captured.** OpenHands resolves its shell without going through OrcaReplay's PATH shim, so `terminal` calls appear in the trace with their output but without the real exit code, duration, or stdout/stderr split. The run prints `warn shell.ineffective` saying so. Record with `--no-shell` if you would rather it claimed nothing than claimed that.

**`egress=blocked` means model-provider egress, not network isolation.** Replay serves model responses from the trace and calls no provider — but it still *executes the recorded tool calls for real*. A `terminal` call that ran `curl` runs it again, over the ordinary network, because that traffic never passes through OrcaReplay. Replay is not a sandbox; to stop all egress, run it inside one.

**The container runtime is not covered.** This works for the SDK used as a library, where OrcaReplay launches the Python process. Recording works by wrapping a process on the host and does not cross a container boundary, so OpenHands running as the packaged product in its own runtime is untested and probably out of reach without passing the proxy into the container.

## Learn more

- [OrcaReplay repository](https://github.com/Continuum-AI-Corp/OrcaReplay) — Apache-2.0
- [The OpenHands integration notes](https://github.com/Continuum-AI-Corp/OrcaReplay/blob/main/docs/integrations.md#openhands) — what is measured, and what is not
- [The check that backs this page](https://github.com/Continuum-AI-Corp/OrcaReplay/tree/main/test/integrations) — records the SDK against a stub origin, kills the origin, replays, and asserts the numbers
