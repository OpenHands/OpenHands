import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

/** Zombies retain a process-group id but cannot hold ports or run cleanup. */
export function groupAlive(
  pgid,
  {
    platform = process.platform,
    signal = process.kill,
    procRoot = "/proc",
    readDirectory = readdirSync,
    readStat = (path) => readFileSync(path, "utf8"),
  } = {},
) {
  if (!Number.isSafeInteger(pgid) || pgid <= 0) return false;
  try {
    signal(-pgid, 0);
  } catch (error) {
    // Permission failures do not establish that a group is dead.
    return error.code !== "ESRCH";
  }
  if (platform !== "linux") return true;
  try {
    let previous;
    for (let scan = 0; scan < 2; scan++) {
      const members = new Set();
      const entries = readDirectory(procRoot).filter((name) =>
        /^\d+$/.test(name),
      );
      for (const name of entries) {
        let stat;
        try {
          stat = readStat(join(procRoot, name, "stat"));
        } catch (error) {
          // A process may exit between enumeration and reading its stat. It
          // cannot keep this group alive; other read failures remain uncertain.
          if (error.code === "ENOENT" || error.code === "ESRCH") continue;
          throw error;
        }
        const close = stat.lastIndexOf(")");
        const fields = stat
          .slice(close + 2)
          .trim()
          .split(/\s+/);
        const processGroup = Number(fields[2]);
        if (
          close < 0 ||
          !stat.startsWith(`${name} (`) ||
          !/^[RSDZTtWXxIPK]$/.test(fields[0]) ||
          !/^-?\d+$/.test(fields[2]) ||
          !Number.isSafeInteger(processGroup)
        ) {
          return true;
        }
        // Linux may report pgrp -1 during exit. Valid nonpositive values cannot
        // belong to our positive target group; malformed metadata stays unsafe.
        if (processGroup !== pgid) continue;
        members.add(name);
        if (fields[0] !== "Z" && fields[0] !== "X" && fields[0] !== "x")
          return true;
      }
      // Require a stable, nonempty dead group in both scans. Ambient /proc
      // churn is irrelevant, but changing target membership or an unseen
      // group after kill(0) is not permission to purge.
      if (
        members.size === 0 ||
        (previous &&
          (members.size !== previous.size ||
            [...members].some((name) => !previous.has(name))))
      )
        return true;
      previous = members;
    }
    return false;
  } catch {
    return true;
  }
}

/** Never delete credentials/state or release a claim while a port remains open. */
export function privatePurgeAllowed(launcherStopped, ports) {
  const states = Object.values(ports);
  return (
    launcherStopped === true &&
    states.length > 0 &&
    states.every((state) => state === "closed")
  );
}
