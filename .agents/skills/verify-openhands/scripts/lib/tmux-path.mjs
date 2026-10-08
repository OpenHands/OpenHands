// Where a run's private tmux server lives (OpenHands/OpenHands#17946: the
// launcher's <stateDir>/tmux is never created, so runs would share one
// server). Under /tmp because a socket below the long run path would exceed
// the 108-byte Unix socket limit; named by a hash of the whole run path so
// two runs whose ids end alike (`qa-run-a1`, `qb-run-a1`) never share one.
import { createHash } from "node:crypto";
import { join, resolve } from "node:path";

export function tmuxPathFor(runDir) {
  const hash = createHash("sha256").update(resolve(runDir)).digest("hex");
  return join("/tmp", `ohv-tmux-${hash.slice(0, 10)}`);
}
