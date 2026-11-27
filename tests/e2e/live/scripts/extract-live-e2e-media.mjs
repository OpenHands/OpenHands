#!/usr/bin/env node

import {
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import {
  dirname,
  extname,
  isAbsolute,
  join,
  relative,
  resolve,
} from "node:path";

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) {
      continue;
    }

    const [rawKey, inlineValue] = arg.slice(2).split("=", 2);
    const key = rawKey.replaceAll("-", "_");
    if (inlineValue !== undefined) {
      args[key] = inlineValue;
      continue;
    }

    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      args[key] = next;
      index += 1;
    } else {
      args[key] = "";
    }
  }
  return args;
}

function assertWithinCwd(label, path) {
  const relativePath = relative(realpathSync(resolve(".")), path);
  if (relativePath.startsWith("..") || isAbsolute(relativePath)) {
    throw new Error(`${label} must be within the current working directory.`);
  }
}

function nearestExistingPath(path) {
  let currentPath = path;
  while (!existsSync(currentPath)) {
    const parentPath = dirname(currentPath);
    if (parentPath === currentPath) {
      throw new Error(`No existing parent found for ${path}`);
    }
    currentPath = parentPath;
  }
  return currentPath;
}

function resolveWithinCwd(label, filePath, options = {}) {
  const resolvedPath = resolve(filePath);
  assertWithinCwd(label, resolvedPath);

  if (existsSync(resolvedPath)) {
    assertWithinCwd(label, realpathSync(resolvedPath));
    return resolvedPath;
  }

  if (options.mustExist) {
    throw new Error(`${label} does not exist: ${resolvedPath}`);
  }

  const existingParent = nearestExistingPath(dirname(resolvedPath));
  const checkedPath = resolve(
    realpathSync(existingParent),
    relative(existingParent, resolvedPath),
  );
  assertWithinCwd(label, checkedPath);
  return resolvedPath;
}

function readJson(path) {
  if (!path || !existsSync(path)) {
    return null;
  }
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

function collectPlaywrightAttachments(results) {
  const attachments = [];

  function visitSuites(suites) {
    for (const suite of suites ?? []) {
      for (const spec of suite.specs ?? []) {
        for (const test of spec.tests ?? []) {
          for (const result of test.results ?? []) {
            attachments.push(...(result.attachments ?? []));
          }
        }
      }
      visitSuites(suite.suites);
    }
  }

  visitSuites(results?.suites);
  return attachments
    .map((attachment) => ({
      contentType: attachment.contentType || "",
      path: resolveExistingMediaPath("attachment", attachment.path || ""),
    }))
    .filter((attachment) => attachment.path);
}

function resolveExistingMediaPath(label, path) {
  if (!path) {
    return "";
  }

  try {
    return resolveWithinCwd(label, path, { mustExist: true });
  } catch {
    return "";
  }
}

function collectFiles(dir) {
  const safeDir = resolveExistingMediaPath("dir", dir);
  if (!safeDir) {
    return [];
  }

  const files = [];
  for (const entry of readdirSync(safeDir)) {
    const path = join(safeDir, entry);
    const stat = lstatSync(path);
    if (stat.isSymbolicLink()) {
      continue;
    }
    if (stat.isDirectory()) {
      files.push(...collectFiles(path));
    } else if (stat.isFile()) {
      files.push(path);
    }
  }
  return files;
}

function isImage(path, contentType = "") {
  return (
    contentType.startsWith("image/") || /\.(gif|jpe?g|png|svg)$/i.test(path)
  );
}

function isVideo(path, contentType = "") {
  return contentType.startsWith("video/") || /\.(mp4|mov|webm)$/i.test(path);
}

function copyMedia(sourcePath, outputDir, targetBase, fallbackExt) {
  if (!sourcePath) {
    return "";
  }

  const safeSourcePath = resolveWithinCwd("media source", sourcePath, {
    mustExist: true,
  });
  mkdirSync(outputDir, { recursive: true });
  const ext = extname(safeSourcePath) || fallbackExt;
  const targetPath = join(outputDir, `${targetBase}${ext}`);
  copyFileSync(safeSourcePath, targetPath);
  return targetPath;
}

function writeOutput(key, value) {
  const outputPath = process.env.GITHUB_OUTPUT;
  if (outputPath) {
    const escaped = String(value).replaceAll("\n", "%0A");
    writeFileSync(outputPath, `${key}=${escaped}\n`, { flag: "a" });
  }
  console.log(`${key}=${value}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const resultsPath = resolveWithinCwd(
    "results",
    args.results || "test-results-live/results.json",
  );
  const testResultsDir = resolveWithinCwd(
    "test-results-dir",
    args.test_results_dir || "test-results-live",
  );
  const outputDir = resolveWithinCwd(
    "output-dir",
    args.output_dir || "test-results-live/media",
  );
  const results = readJson(resultsPath);

  const attachments = collectPlaywrightAttachments(results);
  const files = collectFiles(testResultsDir).map((path) => ({
    contentType: "",
    path,
  }));
  const candidates = [...attachments, ...files];

  const screenshot = candidates.find((candidate) =>
    isImage(candidate.path, candidate.contentType),
  );
  const video = candidates.find((candidate) =>
    isVideo(candidate.path, candidate.contentType),
  );

  const screenshotPath = copyMedia(
    screenshot?.path,
    outputDir,
    "live-agent-response",
    ".png",
  );
  const videoPath = copyMedia(
    video?.path,
    outputDir,
    "live-agent-recording",
    ".webm",
  );

  writeOutput("screenshot_path", screenshotPath);
  writeOutput("video_path", videoPath);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
