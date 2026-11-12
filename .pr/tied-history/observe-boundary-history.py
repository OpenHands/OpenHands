"""Re-run the tied-boundary recipe using only the unchanged Control CLI.

Run after seed-boundary-history.py and Control restart + doctor. All inspection
is read-only DOM/network observation; scrolling/navigation are normal UI actions.
The evidence helper is not a verification-skill driver or expected-result change.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--tied", required=True)
parser.add_argument("--control", required=True)
parser.add_argument("--label", choices=["before", "after"], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cli = args.checkout / ".agents/skills/verify-openhands/scripts/control-openhands"
feature = "F06.load-older-history"

def control(*arguments):
    process = subprocess.run([str(cli), *arguments, "--run", str(args.run)],
                             capture_output=True, text=True, check=False)
    result = json.loads(process.stdout)
    if process.returncode or not result.get("ok"):
        raise RuntimeError(f"Control {arguments[:2]} failed: {result}")
    return result

def observe(kind):
    expression = """(()=>{const m=[...document.querySelectorAll('[data-testid=user-message]')]
      .map(e=>e.textContent.match(/CANVAS-KIND-[0-9]{3}/)?.[0]).filter(Boolean);
      const expected=Array.from({length:100},(_,i)=>'CANVAS-KIND-'+String(i+1).padStart(3,'0'));
      return {count:m.length,unique:new Set(m).size,markers:m,
        missing:expected.filter(x=>!m.includes(x)),
        inSequence:m.every((x,i)=>i===0||m[i-1]<x)};})()""".replace("KIND", kind)
    return control("browser", "eval", expression)["value"]

results = []
for viewport in ["desktop", "phone"]:
    control("browser", "viewport", viewport)
    for attempt in [1, 2]:
        for kind, conversation in [("TIED", args.tied), ("CONTROL", args.control)]:
            control("browser", "network", "--clear")
            control("browser", "goto", "/conversations/" + conversation)
            control("browser", "wait", "text=CANVAS-" + kind + "-100")
            initial = observe(kind)
            if initial["count"] != 50:
                raise RuntimeError(f"expected fresh 50-row initial page: {initial}")
            name = f"{args.label}-{kind.lower()}-{viewport}"
            recording = kind == "TIED" and viewport == "desktop" and attempt == 1
            if recording:
                control("browser", "record", "start", "--feature", feature,
                        "--name", name, "--max-seconds", "90")
                time.sleep(1.2)  # Retain the initial page before scrolling.
            control("browser", "scroll", "testid=chat-scroll-container", "--by", "-100000")
            deadline = time.monotonic() + 15
            while True:
                final = observe(kind)
                if final["count"] > 50:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("older page did not load")
                time.sleep(0.1)
            control("browser", "scroll", "text=CANVAS-" + kind + "-049")
            artifacts = []
            if kind == "TIED" and attempt == 1:
                shot = control("browser", "screenshot", "--feature", feature, "--name", name)
                artifacts.append(Path(shot["path"]).name)
            if recording:
                time.sleep(1.2)
                video = control("browser", "record", "stop")
                artifacts.append(Path(video["path"]).name)
            network = control("browser", "network", "--filter", "events/search", "--last", "10")
            requests = [{"query": r.get("query"), "status": r.get("status")}
                        for r in network["recent"]]
            row = {"viewport": viewport, "attempt": attempt, "kind": kind,
                   "initial": initial, "afterScroll": final,
                   "eventSearchRequests": requests, "artifacts": artifacts}
            results.append(row)
            args.output.write_text(json.dumps({"label": args.label, "checks": results}, indent=2) + "\n")
            print(json.dumps({"viewport": viewport, "attempt": attempt, "kind": kind,
                              "count": final["count"], "missing": final["missing"],
                              "inSequence": final["inSequence"]}), flush=True)
