import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

function linuxProcessGroupState(
  pgid,
  {
    procRoot = "/proc",
    readProcDirectory = () => readdirSync(procRoot),
    readProcStat = (path) => readFileSync(path, "utf8"),
  } = {},
) {
  let entries;
  try {
    entries = readProcDirectory();
  } catch {
    return "unknown";
  }

  let found = false;
  let inspectionUnknown = false;
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) continue;
    let stat;
    try {
      stat = String(readProcStat(join(procRoot, entry, "stat")));
    } catch (error) {
      // Processes can disappear between listing /proc and reading their stat.
      if (["ENOENT", "ESRCH"].includes(error.code)) continue;
      inspectionUnknown = true;
      continue;
    }
    // comm is parenthesized and may itself contain spaces or parentheses, so
    // parse fields after its final closing parenthesis. They begin with
    // state, ppid and pgrp (proc_pid_stat(5) fields 3, 4 and 5).
    const close = stat.lastIndexOf(")");
    if (close < 0) {
      inspectionUnknown = true;
      continue;
    }
    const fields = stat
      .slice(close + 1)
      .trim()
      .split(/\s+/);
    if (fields.length < 3 || !Number.isSafeInteger(Number(fields[2]))) {
      inspectionUnknown = true;
      continue;
    }
    if (Number(fields[2]) !== pgid) continue;
    found = true;
    if (fields[0] !== "Z") return "alive";
  }
  return found && !inspectionUnknown ? "zombie-only" : "unknown";
}

export function groupAlive(
  pgid,
  {
    platform = process.platform,
    signal = process.kill,
    procRoot,
    readProcDirectory,
    readProcStat,
  } = {},
) {
  if (!pgid) return false;
  try {
    signal(-pgid, 0);
  } catch (error) {
    return error.code === "EPERM";
  }

  // kill(2) reports zombie-only process groups as existing on Linux. Other
  // platforms keep the established signal-probe behavior.
  if (platform !== "linux") return true;
  return (
    linuxProcessGroupState(pgid, {
      procRoot,
      readProcDirectory,
      readProcStat,
    }) !== "zombie-only"
  );
}
