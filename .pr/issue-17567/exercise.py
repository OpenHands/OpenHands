"""Drive supported control-openhands commands; retain actual browser receipts."""

import json
import pathlib
import subprocess
import sys

run = pathlib.Path(sys.argv[1]).resolve()
label, viewport, rounds = sys.argv[2], sys.argv[3], int(sys.argv[4])
cli = pathlib.Path(__file__).resolve().parents[2] / ".agents/skills/verify-openhands/scripts/control-openhands"
fixture = json.loads((run / "evidence/fixture.json").read_text())
conversation = fixture["conversationId"]
out = run / "evidence" / "F08.diff-view-modes"
out.mkdir(parents=True, exist_ok=True)
transcript = []
recording = False

def control(*args):
    result = subprocess.run([str(cli), *args, "--run", str(run)], capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    transcript.append({"command": list(args), "result": receipt})
    (out / f"{label}-{viewport}-commands.json").write_text(json.dumps(transcript, indent=2) + "\n")
    if result.returncode or not receipt.get("ok"):
        raise RuntimeError(receipt)
    return receipt

def tab(name):
    target = "testid=conversation-tab-" + name
    if control("browser", "visible", target)["visible"]:
        if control("browser", "attr", target, "aria-selected")["aria-selected"] != "true":
            control("browser", "click", target)
    else:
        control("browser", "click", 'role=button[name="Customize tabs"]')
        control("browser", "click", "testid=conversation-tabs-menu-open-" + name)

def diff(path="src/calc.py"):
    tab("commits")
    control("browser", "wait", "testid=uncommitted-changes-row-toggle")
    if control("browser", "attr", "testid=uncommitted-changes-row-toggle", "aria-expanded")["aria-expanded"] != "true":
        control("browser", "click", "testid=uncommitted-changes-row-toggle")
    row = "testid=uncommitted-changes-row-content >> testid=file-diff-viewer-outer >> has-text=" + path
    if control("browser", "attr", row + " >> testid=collapse", "aria-label")["aria-label"] != "Collapse":
        control("browser", "click", row + " >> testid=collapse")
    expected = "def sub" if path == "src/calc.py" else "Updated README"
    control("browser", "wait-text", expected, "--timeout", "20000")
    text = control("browser", "text", row + " >> testid=editor-container")["text"].replace("\u00a0", " ")
    assert expected in text, text
    return row

try:
    control("browser", "viewport", viewport)
    route = "/conversations/" + conversation + ("/panel" if viewport == "phone" else "")
    control("browser", "goto", route)
    if viewport != "phone":
        control("browser", "click", "testid=right-panel-toggle")
    row = diff()
    control("browser", "errors", "--clear")
    control("browser", "screenshot", "--feature", "F08.diff-view-modes", "--name", label + "-" + viewport + "-diff")
    control("browser", "record", "start", "--feature", "F08.diff-view-modes", "--name", label + "-" + viewport, "--max-seconds", "600")
    recording = True
    for index in range(rounds):
        tab("planner")
        control("browser", "wait-text", "Diff lifecycle verification")
        control("browser", "eval", "window.monaco?.editor.getModels().map(model => ({uri: model.uri.toString(), disposed: model.isDisposed(), text: model.getValue()}))")
        if index == 0:
            control("browser", "screenshot", "--feature", "F08.diff-view-modes", "--name", label + "-" + viewport + "-planner")
        row = diff()
        print(json.dumps({"label": label, "viewport": viewport, "roundTrip": index + 1}), flush=True)
    # Collapse; select another file; switch diff/old/diff; and unmount the drawer.
    control("browser", "click", row + " >> testid=collapse")
    tab("planner")
    control("browser", "wait-text", "Diff lifecycle verification")
    diff()
    diff("README.md")
    tab("planner")
    control("browser", "wait-text", "Diff lifecycle verification")
    row = diff()
    control("browser", "click", row + " >> testid=view-mode-old")
    control("browser", "wait-text", "return a + b")
    control("browser", "click", row + " >> testid=view-mode-diff")
    control("browser", "wait-text", "def sub")
    control("browser", "screenshot", "--feature", "F08.diff-view-modes", "--name", label + "-" + viewport + "-returned")
    if viewport == "phone":
        control("browser", "click", "testid=conversation-mobile-panel-back", "--expect-url", "/conversations/[^/]+$")
    else:
        control("browser", "click", 'role=link[name="New Chat"]', "--expect-url", "/conversations$")
    errors = control("browser", "errors", "--app-only", "--no-warnings")
    control("browser", "eval", "window.monaco?.editor.getModels().map(model => ({uri: model.uri.toString(), disposed: model.isDisposed(), text: model.getValue()}))")
    control("browser", "record", "stop", "--gif")
    recording = False
    error_count = sum("TextModel got disposed before DiffEditorWidget model got reset" in json.dumps(event) for event in errors["events"])
    summary = {"label": label, "viewport": viewport, "roundTrips": rounds, "correctContentsEachReturn": True, "extraCases": ["collapse then Planner", "different file then Planner", "diff/old/diff modes", "route unmount"], "disposedModelErrors": error_count, "browserErrors": errors, "fixture": fixture}
    (out / f"{label}-{viewport}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert (error_count > 0) if label == "baseline" else (error_count == 0), summary
finally:
    if recording:
        control("browser", "record", "stop")
