---
name: thorough_investigation
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
  - investigate
  - diagnose
  - diagnostic
  - debug
  - troubleshoot
  - check
  - verify
  - analyze
  - inspect
  - examine
  - audit
  - health check
  - root cause
  - why
  - broken
  - fails
  - failing
  - doesn't work
  - not working
  - network
  - connectivity
  - performance
  - slow
  - error
  - errors
  - issue
  - issues
  - problem
---

# Thorough Investigation Protocol

When the user asks you to investigate, diagnose, debug, troubleshoot, check, verify,
or analyze anything, follow this protocol. Do **not** finish after 1–2 commands.
Do **not** ask "should I continue?" — just continue.

## Mandatory depth

Before producing any summary or calling `finish`, you MUST:

1. **Collect at least 4 distinct signals.** Examples of signals:
   - current state (`pwd`, `uname -a`, `/etc/os-release`, `uptime`, `date`)
   - network reachability (`ping`, `traceroute`, `curl -I`, `dig`/`nslookup`)
   - open ports & sockets (`ss -ltn`, `ss -tunap`, `lsof -i`)
   - process & resource (`ps auxf`, `top -bn1 | head`, `free -h`, `df -h`)
   - logs (`journalctl`, `/var/log/*`, `dmesg | tail`, application log files)
   - config (`cat` of relevant config, `env | sort`, mounted files)
   - versions (`--version` of relevant tools)

2. **Cross-reference** signals to reach a conclusion; never rely on a single command.

3. **Try multiple angles.** If one command fails or returns nothing useful,
   switch to a different tool (e.g., `ss` → `netstat`; `systemctl` → `ps`;
   `journalctl` → `/var/log/syslog`).

4. **When reporting findings, produce a structured summary** with at minimum:
   - What was checked (bullet list of each signal collected)
   - What the evidence shows (concrete values, not generalities)
   - Conclusion / root cause (or "insufficient data" + what's needed)
   - Next concrete steps (commands or actions), not vague advice

## Do-not patterns

- ❌ `ping 8.8.8.8` → `finish` ("network works")
- ❌ `ls` → `finish` ("looks fine")
- ❌ asking the user which direction to investigate instead of just investigating

## Minimum-effort examples

For a "check the network" request, at least: `ip a`, `ip route`, `cat /etc/resolv.conf`,
`ping -c3 8.8.8.8`, `ping -c3 1.1.1.1`, `dig github.com`, `curl -sI https://github.com`,
`ss -ltn`, then a structured report.

For a "why is X slow" request, at least: reproduce it under `time`, then `top -bn1`,
`free -h`, `iostat` or `vmstat 1 3`, `df -h`, check logs for warnings/errors in the
same timeframe, then a structured report.
