---
name: add_repo_inst
version: 1.0.0
author: openhands
agent: CodeActAgent
triggers:
- /add_repo_inst
inputs:
  - name: REPO_FOLDER_NAME
    description: "Branch for the agent to work on"
---

Please browse the current repository under /workspace/{{ REPO_FOLDER_NAME }}, look at the documentation and relevant code, and understand the purpose of this repository.

Specifically, I want you to create a `.openhands/microagents/repo.md`  file. This file should contain succinct information that summarizes (1) the purpose of this repository, (2) the general setup of this repo, and (3) a brief description of the structure of this repo.

Here's an example:
```markdown
---
name: repo
type: repo
agent: CodeActAgent
---

This repository contains the code for OpenHands, an automated AI software engineer. It has a Python backend
(in the `openhands` directory) and React frontend (in the `frontend` directory).

## General Setup:
To set up the entire repo, including frontend and backend, run `make build`.
You don't need to do this unless the user asks you to, or if you're trying to run the entire application.

Before pushing any changes, you should ensure that any lint errors or simple test errors have been fixed.

* If you've made changes to the backend, you should run `pre-commit run --all-files --config ./dev_config/python/.pre-commit-config.yaml`
* If you've made changes to the frontend, you should run `cd frontend && npm run lint:fix && npm run build ; cd ..`

If either command fails, it may have automatically fixed some issues. You should fix any issues that weren't automatically fixed,
then re-run the command to ensure it passes.

## Repository Structure
Backend:
- Located in the `openhands` directory
- Testing:
  - All tests are in `tests/unit/test_*.py`
  - To test new code, run `poetry run pytest tests/unit/test_xxx.py` where `xxx` is the appropriate file for the current functionality
  - Write all tests with pytest

Frontend:
- Located in the `frontend` directory
- Prerequisites: A recent version of NodeJS / NPM
- Setup: Run `npm install` in the frontend directory
- Testing:
  - Run tests: `npm run test`
  - To run specific tests: `npm run test -- -t "TestName"`
- Building:
  - Build for production: `npm run build`
- Environment Variables:
  - Set in `frontend/.env` or as environment variables
  - Available variables: VITE_BACKEND_HOST, VITE_USE_TLS, VITE_INSECURE_SKIP_VERIFY, VITE_FRONTEND_PORT
- Internationalization:
  - Generate i18n declaration file: `npm run make-i18n`
```

Now, please write a similar markdown for the current repository.
Read all the GitHub workflows under .github/ of the repository (if this folder exists) to understand the CI checks (e.g., linter, pre-commit), and include those in the repo.md file.
