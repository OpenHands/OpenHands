# Before/after: `<RUNTIME_SERVICES>` on agent-profile launches

Full local stack (`npm run dev`: ingress, automation and Vite, which advertise runtime services), with the pinned agent-server 1.50.1 and the recording mock LLM. State was isolated in a scratch dir. Playwright: activate a **named** OpenHands profile, send "Create an automation that runs every morning." from the home chat, then open **Agent Tools & Metadata → System Message**.

```
[before] launched via agent_profile_id | agent_launch_additions sent: false | <RUNTIME_SERVICES> reached the LLM: false | System Message shows it: false
[after]  launched via agent_profile_id | agent_launch_additions sent: true  | <RUNTIME_SERVICES> reached the LLM: true  | System Message shows it: true
```

- `before-main.png`: `origin/main`. The dynamic block goes straight from `</SKILLS>` to `<CUSTOM_SECRETS>`, with no `<RUNTIME_SERVICES>`.
- `after-this-pr.png`: this branch. The `<RUNTIME_SERVICES>` block lists the local automation backend.
