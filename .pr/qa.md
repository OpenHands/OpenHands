# Local QA for issue #14153

## Scenario

Verified the local OpenHands GUI can:
1. open the Integrations settings page,
2. save a valid GitHub token, and
3. disconnect the saved token.

## Environment

- Local app run via `make build && make run FRONTEND_PORT=12000 FRONTEND_HOST=0.0.0.0 BACKEND_HOST=0.0.0.0`
- Browser automation targeted `https://work-1-crbjmclxljyvaakn.prod-runtime.all-hands.dev/settings/integrations`

## Artifacts

- Animated GIF: `.pr/artifacts/integrations-local-gif/integrations-local-qa.gif`
- Browser frames and result JSON are in `.pr/artifacts/integrations-local-gif/`
