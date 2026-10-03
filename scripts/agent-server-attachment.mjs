import { readFileSync } from "node:fs";
import {
  ServerClient,
  SettingsClient,
  compareAgentServerVersions,
} from "@openhands/typescript-client/clients";

// @spec LA-001 — Explicit attachment preserves server ownership
export function readExistingSessionApiKey(env, persistedKeyPath) {
  if (env.LOCAL_BACKEND_API_KEY?.trim()) {
    return env.LOCAL_BACKEND_API_KEY;
  }
  try {
    const key = readFileSync(persistedKeyPath, "utf8").trim();
    if (key) return key;
  } catch {
    // Attaching must never generate or overwrite the external server's key.
  }
  throw new Error(
    "Attaching requires the existing Agent Server's session key. Set " +
      "LOCAL_BACKEND_API_KEY or point OH_SESSION_API_KEY_PATH at its key file.",
  );
}

// @spec LA-002 — Verify an attached server before starting services
export async function validateAttachedAgentServer(config, minimumVersion) {
  const options = {
    host: config.backendBaseUrl,
    apiKey: config.sessionApiKey,
    timeout: 5000,
  };
  const server = new ServerClient(options);
  const settings = new SettingsClient(options);
  let info;
  try {
    await server.getAlive();
    // /alive and /server_info are public. Authenticate through the same
    // read-only settings endpoint as the browser's backend connection check.
    await settings.getSettings();
    info = await server.getServerInfo();
  } catch (error) {
    const detail =
      error?.status === 401 || error?.status === 403
        ? "The configured session key was rejected."
        : "Health, authentication or server-info verification failed.";
    // Do not include response bodies or transport errors that may contain
    // credentials or details from an unrelated service on the configured port.
    throw new Error(
      `Cannot attach to Agent Server at ${config.backendBaseUrl}. ${detail}`,
    );
  } finally {
    server.close();
    settings.close();
  }

  const version = info?.version || info?.sdk_version;
  const comparison =
    typeof version === "string"
      ? compareAgentServerVersions(version, minimumVersion)
      : null;
  if (comparison === null || comparison < 0) {
    throw new Error(
      `Cannot attach to Agent Server at ${config.backendBaseUrl}: ` +
        `a valid version >= ${minimumVersion} is required.`,
    );
  }
}
