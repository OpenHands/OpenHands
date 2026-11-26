import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";

/**
 * @param {string} packageName
 * @param {string} entryPath
 * @returns {string}
 */
export function resolvePackageVersion(packageName, entryPath) {
  let directory = dirname(entryPath);

  while (true) {
    try {
      const manifest = JSON.parse(
        readFileSync(join(directory, "package.json"), "utf8"),
      );
      if (
        manifest.name === packageName &&
        typeof manifest.version === "string"
      ) {
        return manifest.version;
      }
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }

    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error(`Could not find package.json for ${packageName}`);
    }
    directory = parent;
  }
}
