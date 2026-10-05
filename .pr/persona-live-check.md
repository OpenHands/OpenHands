# Persona / instructions live check

- Date: 2026-10-05
- Canvas: `c9fd7fd11` (#17843 stacked on #17516), `npm run dev`, fresh state dir, agent-server run with an isolated `HOME`
- Agent-server: 1.53.0 (`/server_info`), browser tool usable
- LLM: recording mock (`openai/mock-test-model`); facts below come from its `GET /admin/requests`
- Launches: each profile picked in the home chat profile menu (`agent_profile_id` path)

| Profile | Mode | Starts with persona | Kept sections sent | Persona-layer sections sent | Instructions sent |
|---|---|---|---|---|---|
| `explorer` | Custom persona | true | MEMORY, SECURITY, SECURITY_RISK_ASSESSMENT, BROWSER_TOOLS, EXTERNAL_SERVICES, PROCESS_MANAGEMENT, SKILLS | none (no SOUL, ROLE, EFFICIENCY, FILE_SYSTEM_GUIDELINES, CODE_QUALITY, VERSION_CONTROL, PULL_REQUESTS, PROBLEM_SOLVING_WORKFLOW, SELF_DOCUMENTATION, ENVIRONMENT_SETUP, TROUBLESHOOTING) | n/a |
| `helper` | Default + your instructions | false | all of the above | SOUL, ROLE, EFFICIENCY, … TROUBLESHOOTING (full default set) | true ("Always answer in bullet points." after `</SKILLS>`) |
| `plain` | OpenHands default | false | all of the above | full default set | n/a |

All three conversations finished with the mock's reply.

Page errors: none. The only console error is a 404 from `GET /api/file/search_subdirs?path=/projects` (the workspace picker probing a missing folder; unrelated to this PR).
