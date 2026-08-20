import type { Config } from "@react-router/dev/config";
import { vercelPreset } from "@vercel/react-router/vite";

const normalizeBasePath = (value?: string) => {
  const raw = value?.trim();
  if (!raw || raw === "/") return undefined;

  const withLeadingSlash = raw.startsWith("/") ? raw : `/${raw}`;
  return withLeadingSlash.replace(/\/+$/, "");
};

const basename = normalizeBasePath(process.env.VITE_BASE_PATH);

/**
 * This script is used to unpack the client directory from the frontend build directory.
 * Remix SPA mode builds the client directory into the build directory. This function
 * moves the contents of the client directory to the build directory and then removes the
 * client directory.
 *
 * This script is used in the buildEnd function of the Vite config.
 */
let unpackClientDirectoryPromise: Promise<void> | null = null;

const moveBuildEntry = async (
  fs: typeof import("fs"),
  source: string,
  destination: string,
) => {
  await fs.promises.rm(destination, { recursive: true, force: true });

  try {
    await fs.promises.rename(source, destination);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EPERM" && code !== "EXDEV") {
      throw error;
    }

    await fs.promises.cp(source, destination, {
      recursive: true,
      force: true,
    });
    await fs.promises.rm(source, { recursive: true, force: true });
  }
};

const unpackClientDirectoryOnce = async () => {
  if (process.env.VERCEL) {
    // Vercel's React Router builder reads static assets from build/client.
    return;
  }

  const fs = await import("fs");
  const path = await import("path");

  const buildDir = path.resolve(__dirname, "build");
  const clientDir = path.resolve(buildDir, "client");

  let files: string[];
  try {
    files = await fs.promises.readdir(clientDir);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return;
    }
    throw error;
  }

  for (const file of files) {
    await moveBuildEntry(
      fs,
      path.resolve(clientDir, file),
      path.resolve(buildDir, file),
    );
  }

  await fs.promises.rm(clientDir, { recursive: true, force: true });
};

const unpackClientDirectory = async () => {
  unpackClientDirectoryPromise ??= unpackClientDirectoryOnce().finally(() => {
    unpackClientDirectoryPromise = null;
  });

  await unpackClientDirectoryPromise;
};

export default {
  appDirectory: "src",
  ...(basename ? { basename } : {}),
  buildEnd: unpackClientDirectory,
  presets: [vercelPreset()],
  ssr: false,
} satisfies Config;
