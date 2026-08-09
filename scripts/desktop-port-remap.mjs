/**
 * Desktop port remapping helpers.
 *
 * When the Electron app starts and preferred service ports are already in use,
 * we propose free alternatives and (if the user accepts) apply them via the
 * same env vars that `dev-with-automation.mjs` / `buildConfig` already honor.
 */

/** @typedef {{ name: string, label: string, preferred: number, envKey: string }} DesktopPortSpec */

/**
 * Build the preferred desktop port list from defaults + optional env overrides.
 *
 * @param {{ ports: { proxy: number, agentServer: number, automation: number } }} defaults
 * @param {NodeJS.ProcessEnv} [env]
 * @returns {DesktopPortSpec[]}
 */
export function getDesktopPreferredPorts(defaults, env = process.env) {
  const parseOr = (raw, fallback) => {
    const n = parseInt(String(raw ?? ""), 10);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };

  return [
    {
      name: "ingress",
      label: "UI / ingress",
      preferred: parseOr(env.PORT, defaults.ports.proxy),
      envKey: "PORT",
    },
    {
      name: "agent-server",
      label: "agent-server",
      preferred: parseOr(
        env.OH_CANVAS_SAFE_BACKEND_PORT,
        defaults.ports.agentServer,
      ),
      envKey: "OH_CANVAS_SAFE_BACKEND_PORT",
    },
    {
      name: "automation",
      label: "automation",
      preferred: parseOr(
        env.OH_CANVAS_SAFE_AUTOMATION_PORT,
        defaults.ports.automation,
      ),
      envKey: "OH_CANVAS_SAFE_AUTOMATION_PORT",
    },
    {
      name: "frontend",
      label: "static frontend",
      preferred: parseOr(env.OH_CANVAS_SAFE_VITE_PORT, 3001),
      envKey: "OH_CANVAS_SAFE_VITE_PORT",
    },
  ];
}

/**
 * Compare preferred vs allocated ports and build env overrides for remaps.
 *
 * @param {DesktopPortSpec[]} preferredPorts
 * @param {Record<string, number>} allocated - name → free port from findFreePorts
 * @returns {{ remaps: Array<{ name: string, label: string, preferred: number, actual: number }>, env: Record<string, string> }}
 */
export function buildDesktopPortOverrides(preferredPorts, allocated) {
  const remaps = [];
  /** @type {Record<string, string>} */
  const env = {};

  for (const spec of preferredPorts) {
    const actual = allocated[spec.name];
    if (typeof actual !== "number" || actual <= 0) {
      throw new Error(
        `Missing allocated port for desktop service "${spec.name}"`,
      );
    }
    if (actual !== spec.preferred) {
      remaps.push({
        name: spec.name,
        label: spec.label,
        preferred: spec.preferred,
        actual,
      });
      env[spec.envKey] = String(actual);
    }
  }

  return { remaps, env };
}

/**
 * Human-readable detail for the "ports in use" confirmation dialog.
 *
 * @param {Array<{ label: string, preferred: number, actual: number }>} remaps
 * @returns {string}
 */
export function formatPortRemapMessage(remaps) {
  const lines = remaps.map(
    ({ label, preferred, actual }) =>
      `• ${label}: ${preferred} → ${actual}`,
  );
  return (
    "These default ports are already in use (another Agent Canvas instance " +
    "or a different app may be running):\n\n" +
    `${lines.join("\n")}\n\n` +
    "Use the free alternate ports listed above and continue starting?"
  );
}

/**
 * Resolve preferred desktop ports to free ones. Caller shows a confirm dialog
 * when `remaps` is non-empty before applying `env` to `process.env`.
 *
 * @param {DesktopPortSpec[]} preferredPorts
 * @param {(configs: Array<{name: string, preferred: number}>) => Promise<Record<string, number>>} findFreePorts
 * @returns {Promise<{ remaps: Array<{ name: string, label: string, preferred: number, actual: number }>, env: Record<string, string>, allocated: Record<string, number> }>}
 */
export async function planDesktopPortRemap(preferredPorts, findFreePorts) {
  const allocated = await findFreePorts(
    preferredPorts.map(({ name, preferred }) => ({ name, preferred })),
  );
  const { remaps, env } = buildDesktopPortOverrides(preferredPorts, allocated);
  return { remaps, env, allocated };
}
