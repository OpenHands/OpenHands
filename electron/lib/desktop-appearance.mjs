import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export const APPEARANCE_FILE = "desktop-appearance.json";

/**
 * Load the persisted appearance ("dark" | "light") from the user data dir.
 * Defaults to "dark".
 *
 * @param {string | null | undefined} userDataPath
 * @returns {"dark" | "light"}
 */
export function loadStoredAppearance(userDataPath) {
  if (!userDataPath) return "dark";
  try {
    const raw = readFileSync(join(userDataPath, APPEARANCE_FILE), "utf-8");
    const data = JSON.parse(raw);
    if (data?.appearance === "light" || data?.appearance === "dark") {
      return data.appearance;
    }
  } catch {
    // Missing or invalid file defaults to dark
  }
  return "dark";
}

/**
 * Persist the appearance ("dark" | "light") to the user data dir.
 *
 * @param {string | null | undefined} userDataPath
 * @param {"dark" | "light"} appearance
 */
export function persistAppearance(userDataPath, appearance) {
  if (!userDataPath) return;
  if (appearance !== "light" && appearance !== "dark") return;
  try {
    writeFileSync(
      join(userDataPath, APPEARANCE_FILE),
      JSON.stringify({ appearance }, null, 2),
      "utf-8",
    );
  } catch {
    // Best-effort write failure is safely ignored
  }
}
