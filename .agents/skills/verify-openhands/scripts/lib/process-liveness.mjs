import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, readdirSync } from "node:fs";

export function isZombieStat(stat) {
  if (!stat || typeof stat !== "string") return false;
  return stat.trim().toUpperCase().startsWith("Z");
}

export function parseProcStat(content) {
  if (!content) return null;
  const lastParen = content.lastIndexOf(")");
  if (lastParen === -1) return null;
  const rest = content.slice(lastParen + 1).trim();
  const fields = rest.split(/\s+/);
  if (fields.length < 3) return null;
  return {
    state: fields[0],
    ppid: Number(fields[1]),
    pgid: Number(fields[2]),
  };
}

export function inspectPidWithProc(pid, readProcStat = defaultReadProcStat) {
  const content = readProcStat(pid);
  if (!content) return null;
  const parsed = parseProcStat(content);
  return parsed ? parsed.state : null;
}

export function inspectPidWithPs(pid, runPs = defaultRunPs) {
  // First try specific -p <pid>
  let res = runPs(["-o", "stat=", "-p", String(pid)]);
  if (res && res.status === 0 && res.stdout && res.stdout.trim()) {
    return res.stdout.trim();
  }
  // Try checking from full process table if available
  res = runPs(["-eo", "pid=,pgid=,stat="]);
  if (res && res.status === 0 && res.stdout && res.stdout.trim()) {
    const lines = res.stdout.trim().split("\n");
    for (const line of lines) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 3 && Number(parts[0]) === pid) {
        return parts[2];
      }
    }
  }
  return null;
}

export function inspectGroupWithPs(pgid, runPs = defaultRunPs) {
  const res = runPs(["-eo", "pid=,pgid=,stat="]);
  if (!res || res.status !== 0 || !res.stdout) return null;
  const lines = res.stdout.trim().split("\n");
  const members = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 3) {
      const memberPgid = Number(parts[1]);
      if (memberPgid === pgid) {
        members.push({ pid: Number(parts[0]), stat: parts[2] });
      }
    }
  }
  return members;
}

export function inspectGroupWithProc(
  pgid,
  readProcPids = defaultReadProcPids,
  readProcStat = defaultReadProcStat,
) {
  const pids = readProcPids();
  if (!pids) return null;
  const members = [];
  for (const pid of pids) {
    const content = readProcStat(pid);
    if (!content) continue;
    const parsed = parseProcStat(content);
    if (parsed && parsed.pgid === pgid) {
      members.push({ pid: Number(pid), stat: parsed.state });
    }
  }
  return members;
}

function defaultReadProcStat(pid) {
  try {
    const statPath = `/proc/${pid}/stat`;
    if (existsSync(statPath)) {
      return readFileSync(statPath, "utf8");
    }
  } catch {
    // not readable or gone
  }
  return null;
}

function defaultReadProcPids() {
  try {
    if (!existsSync("/proc")) return null;
    return readdirSync("/proc").filter((e) => /^\d+$/.test(e));
  } catch {
    return null;
  }
}

function defaultRunPs(args) {
  try {
    return spawnSync("ps", args, { encoding: "utf8" });
  } catch {
    return null;
  }
}

export function createLivenessCheckers({
  platform = process.platform,
  kill = process.kill,
  runPs = defaultRunPs,
  readProcStat = defaultReadProcStat,
  readProcPids = defaultReadProcPids,
} = {}) {
  function processAlive(pid) {
    if (!pid) return false;
    try {
      kill(pid, 0);
    } catch (error) {
      return error.code === "EPERM";
    }
    if (platform !== "linux") return true;

    const procState = inspectPidWithProc(pid, readProcStat);
    if (procState !== null) {
      return !isZombieStat(procState);
    }

    const psState = inspectPidWithPs(pid, runPs);
    if (psState !== null) {
      return !isZombieStat(psState);
    }

    try {
      kill(pid, 0);
    } catch (error) {
      if (error.code !== "EPERM") return false;
    }

    return true;
  }

  function groupAlive(pgid) {
    if (!pgid) return false;
    try {
      kill(-pgid, 0);
    } catch (error) {
      return error.code === "EPERM";
    }
    if (platform !== "linux") return true;

    let members = inspectGroupWithPs(pgid, runPs);
    if (!members || members.length === 0) {
      const procMembers = inspectGroupWithProc(
        pgid,
        readProcPids,
        readProcStat,
      );
      if (procMembers && procMembers.length > 0) {
        members = procMembers;
      }
    }

    if (members && members.length > 0) {
      const allZombies = members.every((m) => isZombieStat(m.stat));
      return !allZombies;
    }

    try {
      kill(-pgid, 0);
    } catch (error) {
      if (error.code !== "EPERM") return false;
    }

    return true;
  }

  return { processAlive, groupAlive };
}

const defaultCheckers = createLivenessCheckers();
export const processAlive = defaultCheckers.processAlive;
export const groupAlive = defaultCheckers.groupAlive;
