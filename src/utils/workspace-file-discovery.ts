export interface WorkspaceFileDiscovery {
  excludedPatterns: string[];
  /** Zero means unlimited. */
  maxFiles: number;
  includeSymlinks: boolean;
}

export const DEFAULT_FILE_DISCOVERY: WorkspaceFileDiscovery = {
  excludedPatterns: [
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".turbo",
    ".parcel-cache",
    "target",
  ],
  maxFiles: 2000,
  includeSymlinks: false,
};

export function isValidFileLimit(value: number): boolean {
  return (
    Number.isSafeInteger(value) && value >= 0 && value < Number.MAX_SAFE_INTEGER
  );
}

export function normalizeFileDiscovery(value: unknown): WorkspaceFileDiscovery {
  if (!value || typeof value !== "object") return DEFAULT_FILE_DISCOVERY;
  const options = value as Partial<WorkspaceFileDiscovery>;
  return {
    excludedPatterns:
      Array.isArray(options.excludedPatterns) &&
      options.excludedPatterns.every(
        (pattern) =>
          typeof pattern === "string" &&
          pattern.length > 0 &&
          !/[\0\r\n]/.test(pattern),
      )
        ? options.excludedPatterns
        : DEFAULT_FILE_DISCOVERY.excludedPatterns,
    maxFiles:
      typeof options.maxFiles === "number" && isValidFileLimit(options.maxFiles)
        ? options.maxFiles
        : DEFAULT_FILE_DISCOVERY.maxFiles,
    includeSymlinks: options.includeSymlinks === true,
  };
}

const shellQuote = (value: string) => `'${value.replaceAll("'", "'\\''")}'`;

// @spec WFD-001 — Configurable local workspace discovery
export function buildWorkspaceFileListCommand(
  options: WorkspaceFileDiscovery,
): string {
  const patterns = options.excludedPatterns.map((pattern) => {
    if (!pattern.includes("/")) return `-name ${shellQuote(pattern)}`;
    const relative = pattern.startsWith("./") ? pattern : `./${pattern}`;
    return `-path ${shellQuote(relative)}`;
  });
  const prune = patterns.length
    ? `-type d \\( ${patterns.join(" -o ")} \\) -prune -o `
    : "";
  const types = options.includeSymlinks
    ? "\\( -type f -o \\( -type l -exec test -f {} \\; \\) \\)"
    : "-type f";
  // Fetch one extra path to distinguish truncation from an exact-size result.
  const limit =
    options.maxFiles > 0 ? ` | head -n ${options.maxFiles + 1}` : "";
  return `find . ${prune}${types} -print 2>/dev/null | sort${limit}`;
}

export function parseWorkspaceFileList(stdout: string, maxFiles: number) {
  const paths = Array.from(
    new Set(
      stdout
        .split(/\r?\n/)
        .filter(Boolean)
        .map((path) => (path.startsWith("./") ? path.slice(2) : path)),
    ),
  );
  return {
    paths: maxFiles > 0 ? paths.slice(0, maxFiles) : paths,
    isTruncated: maxFiles > 0 && paths.length > maxFiles,
  };
}

// A Windows-hosted local backend runs commands through cmd.exe, where `find`
// is the text-search tool and `head` and `/dev/null` do not exist, so the POSIX
// pipeline exits non-zero and the Files tab shows its empty state. `dir /s`
// prints absolute backslash paths; findstr drops excluded directories and the
// caller makes the rest relative with `parseWindowsWorkspaceFileList`.
const isCmdSafePattern = (pattern: string) => !/["%^&|<>!*?]/.test(pattern);

export function buildWindowsWorkspaceFileListCommand(
  options: WorkspaceFileDiscovery,
  workingDir: string,
): string {
  const root = workingDir.replace(/[\\/]+$/, "").toLowerCase();
  const excludes = options.excludedPatterns
    .map((pattern) => pattern.replace(/^\.\//, "").replaceAll("/", "\\"))
    .filter(
      (pattern) =>
        isCmdSafePattern(pattern) &&
        // A pattern that already appears in the workspace root path would drop
        // every file; the parser filters those on the relative path instead.
        !`\\${root.replaceAll("/", "\\")}\\`.includes(
          `\\${pattern.toLowerCase()}\\`,
        ),
    )
    .map((pattern) => `/c:"\\${pattern}\\"`);
  const filter = excludes.length
    ? ` | findstr /v /i ${excludes.join(" ")}`
    : "";
  return `dir /b /s /a:-d 2>nul${filter} | sort`;
}

export function parseWindowsWorkspaceFileList(
  stdout: string,
  workingDir: string,
  options: WorkspaceFileDiscovery,
) {
  const root = workingDir.replaceAll("/", "\\").replace(/\\+$/, "");
  const excluded = options.excludedPatterns
    .filter((pattern) => !pattern.includes("/"))
    .map((pattern) => pattern.toLowerCase());
  const relative = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) =>
      line.toLowerCase().startsWith(`${root.toLowerCase()}\\`)
        ? line.slice(root.length + 1)
        : line,
    )
    .map((line) => line.replace(/\\/g, "/"))
    .filter((line) => {
      const dirs = line.toLowerCase().split("/").slice(0, -1);
      return !dirs.some((dir) => excluded.includes(dir));
    });
  return parseWorkspaceFileList(relative.join("\n"), options.maxFiles);
}
