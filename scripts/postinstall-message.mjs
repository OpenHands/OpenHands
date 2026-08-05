const GREEN = "\x1b[32m";
const CYAN = "\x1b[36m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";

const message = `
${GREEN}${BOLD}✓ @openhands/agent-canvas installed successfully!${RESET}

To start Agent Canvas, run:

  ${CYAN}${BOLD}agent-canvas${RESET}

This launches the full stack (agent-server + UI) at ${CYAN}http://localhost:8000${RESET}.

Other useful commands:
  ${CYAN}agent-canvas --help${RESET}      Show all available options
  ${CYAN}agent-canvas --port 3000${RESET} Start on a custom port
  ${CYAN}agent-canvas --info${RESET}      Show version and stack info

Docs: https://docs.openhands.dev/openhands/usage/agent-canvas/setup
`;

console.log(message);
