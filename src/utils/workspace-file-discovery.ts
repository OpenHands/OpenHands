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
