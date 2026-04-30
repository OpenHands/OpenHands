import type { Config } from "@react-router/dev/config";

/**
 * This script is used to unpack the client directory from the frontend build directory.
 * Remix SPA mode builds the client directory into the build directory. This function
 * moves the contents of the client directory to the build directory and then removes the
 * client directory.
 *
 * This script is used in the buildEnd function of the Vite config.
 */
const unpackClientDirectory = async () => {
  const fs = await import("fs");
  const path = await import("path");

  const movePath = async (fromPath: string, toPath: string) => {
    try {
      await fs.promises.rm(toPath, { recursive: true, force: true });
      await fs.promises.rename(fromPath, toPath);
      return;
    } catch (error) {
      const code =
        error && typeof error === "object" && "code" in error
          ? String(error.code)
          : "";

      // Windows can transiently lock directories during build output indexing.
      if (code !== "EPERM" && code !== "EXDEV") {
        throw error;
      }
    }

    const fromStat = await fs.promises.stat(fromPath);
    await fs.promises.rm(toPath, { recursive: true, force: true });

    if (fromStat.isDirectory()) {
      await fs.promises.cp(fromPath, toPath, {
        recursive: true,
        force: true,
        errorOnExist: false,
      });
      await fs.promises.rm(fromPath, { recursive: true, force: true });
      return;
    }

    await fs.promises.copyFile(fromPath, toPath);
    await fs.promises.rm(fromPath, { force: true });
  };

  const buildDir = path.resolve(__dirname, "build");
  const clientDir = path.resolve(buildDir, "client");

  const files = await fs.promises.readdir(clientDir);
  // Sequential renames: parallel renames on Windows can fail with EPERM (AV/indexer locks).
  /* eslint-disable no-await-in-loop -- must rename one file at a time on Windows */
  for (const file of files) {
    await movePath(path.resolve(clientDir, file), path.resolve(buildDir, file));
  }
  /* eslint-enable no-await-in-loop */

  await fs.promises.rm(clientDir, { recursive: true, force: true });
};

export default {
  appDirectory: "src",
  buildEnd: unpackClientDirectory,
  ssr: false,
} satisfies Config;
