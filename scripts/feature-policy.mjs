/**
 * Resolve the operator's startup feature policy.
 *
 * The policy is a JSON document `{ "version": 1, "disabled": ["<id>", …] }`
 * supplied inline through AGENT_CANVAS_FEATURES or as a file path through
 * AGENT_CANVAS_FEATURES_FILE. Every feature is on unless listed. Known IDs live
 * in config/features.json.
 *
 * Invalid configuration throws: an operator who asked for a locked-down
 * instance must never get the full one because of a typo.
 *
 * Docker copies scripts flat into /opt/agent-canvas and config/features.json
 * to /opt/config, so the `../config` lookup below holds in every layout.
 */

import { readFileSync } from "node:fs";

import featureCatalog from "../config/features.json" with { type: "json" };

export const FEATURES_ENV = "AGENT_CANVAS_FEATURES";
export const FEATURES_FILE_ENV = "AGENT_CANVAS_FEATURES_FILE";
export const FEATURE_POLICY_VERSION = 1;

export function loadFeatureCatalog() {
  return new Set(featureCatalog.features.map((feature) => feature.id));
}

function parsePolicyJson(raw, source) {
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `Invalid feature policy JSON in ${source}: ${err instanceof Error ? err.message : err}`,
    );
  }
}

/**
 * @param {NodeJS.ProcessEnv} env
 * @param {Set<string>} [knownIds]
 * @returns {{ version: 1, disabled: string[] } | null} null when no policy is set
 */
export function resolveFeaturePolicy(env, knownIds = loadFeatureCatalog()) {
  const inline = env[FEATURES_ENV]?.trim();
  const file = env[FEATURES_FILE_ENV]?.trim();

  if (inline && file) {
    throw new Error(`Set ${FEATURES_ENV} or ${FEATURES_FILE_ENV}, not both.`);
  }
  if (!inline && !file) return null;

  let raw = inline;
  let source = FEATURES_ENV;
  if (file) {
    source = `${FEATURES_FILE_ENV} (${file})`;
    try {
      raw = readFileSync(file, "utf8");
    } catch (err) {
      throw new Error(
        `Cannot read feature policy file ${file}: ${err instanceof Error ? err.message : err}`,
      );
    }
  }

  const policy = parsePolicyJson(raw, source);
  if (!policy || typeof policy !== "object" || Array.isArray(policy)) {
    throw new Error(`Feature policy in ${source} must be a JSON object.`);
  }
  if (policy.version !== FEATURE_POLICY_VERSION) {
    throw new Error(
      `Unsupported feature policy version in ${source}: ${JSON.stringify(policy.version)} (expected ${FEATURE_POLICY_VERSION}).`,
    );
  }
  const disabled = policy.disabled ?? [];
  if (
    !Array.isArray(disabled) ||
    disabled.some((id) => typeof id !== "string")
  ) {
    throw new Error(
      `Feature policy "disabled" in ${source} must be an array of feature IDs.`,
    );
  }
  const unknown = disabled.filter((id) => !knownIds.has(id));
  if (unknown.length > 0) {
    throw new Error(
      `Unknown feature IDs in ${source}: ${unknown.join(", ")}. Known IDs: ${[...knownIds].join(", ")}.`,
    );
  }

  return { version: FEATURE_POLICY_VERSION, disabled: [...new Set(disabled)] };
}
