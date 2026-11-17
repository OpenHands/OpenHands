"""An agent outside Canvas, holding only the Canvas session key."""
import os, sys, threading
from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.workspace import RemoteWorkspace

host = f"http://127.0.0.1:{os.environ['PROXY_PORT']}/mars/sessions/sess_e2e"
key = os.environ["CANVAS_KEY"]

# 1. Wrong key: REST refused at the proxy.
bad = RemoteWorkspace(host=host, api_key="wrong-key", working_dir="/tmp")
refused = bad.execute_command("echo should-not-run")
if "should-not-run" in (refused.stdout or ""):
    print("WRONG KEY: unexpectedly allowed"); sys.exit(1)
print("wrong key refused: exit", refused.exit_code, "|", (refused.stderr or "").strip()[:90])

# 2. Right key: a terminal command in the "session".
ws = RemoteWorkspace(host=host, api_key=key, working_dir=os.environ["GUEST_DIR"])
result = ws.execute_command("echo hello-from-the-session && whoami")
print("terminal:", result.exit_code, result.stdout.strip().replace("\n", " | "))

# 3. A conversation whose event stream logs in by its first message.
agent = Agent(llm=LLM(model="openai/gpt-4o-mini", api_key="sk-not-used", usage_id="e2e"), tools=[])
events = []
conversation = Conversation(agent=agent, workspace=ws, callbacks=[lambda e: events.append(type(e).__name__)])
print("conversation:", conversation.id, "| events so far:", sorted(set(events)))
conversation.close()
print("done")
