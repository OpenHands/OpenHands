// Map a workspace-file path to a Prism grammar name (or `null` when we
// don't have a registered grammar for it, in which case the caller should
// fall back to plain pre/text rendering).
//
// We rely on the language allowlist already registered in
// `src/components/features/markdown/syntax-highlighter.ts` — every entry
// here MUST resolve to either a language name or an alias that file
// registers, otherwise PrismLight will fall back to no highlighting at
// all (and log a warning).
//
// Two lookup keys are tried, in order:
//   1. The lowercased final-segment extension (`tsx`, `py`, `yaml`, …).
//   2. A few well-known no-extension filenames (`Dockerfile`, `Makefile`).
// We deliberately keep this list tight: every entry maps a real file
// users see in workspaces to a grammar Prism actually understands.
const EXTENSION_TO_LANGUAGE: Record<string, string> = {
  // JS / TS family
  js: "javascript",
  jsx: "jsx",
  mjs: "javascript",
  cjs: "javascript",
  ts: "typescript",
  tsx: "tsx",

  // Python
  py: "python",
  pyi: "python",
  pyw: "python",

  // Web
  html: "markup",
  htm: "markup",
  svg: "markup",
  xml: "markup",
  css: "css",
  scss: "scss",
  sass: "sass",
  less: "less",

  // Data / config
  json: "json",
  json5: "json5",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  ini: "ini",
  cfg: "ini",
  properties: "properties",
  conf: "ini",
  env: "bash",

  // Markdown / docs
  md: "markdown",
  markdown: "markdown",
  mdx: "markdown",

  // Shell / scripts
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  fish: "bash",
  bat: "batch",
  cmd: "batch",
  ps1: "powershell",
  ps: "powershell",

  // Systems
  c: "c",
  h: "c",
  cpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  hpp: "cpp",
  hh: "cpp",
  cs: "csharp",
  java: "java",
  kt: "kotlin",
  kts: "kotlin",
  scala: "scala",
  rs: "rust",
  go: "go",
  rb: "ruby",
  php: "php",
  swift: "swift",
  m: "objectivec",
  mm: "objectivec",

  // Functional
  hs: "haskell",
  ex: "elixir",
  exs: "elixir",
  erl: "erlang",
  fs: "fsharp",
  fsx: "fsharp",
  ml: "ocaml",
  mli: "ocaml",
  clj: "clojure",
  cljs: "clojure",
  cljc: "clojure",

  // Scripting / data
  lua: "lua",
  pl: "perl",
  pm: "perl",
  r: "r",
  jl: "julia",
  dart: "dart",
  groovy: "groovy",
  gradle: "groovy",

  // Infrastructure
  tf: "hcl",
  hcl: "hcl",
  dockerfile: "docker",

  // Misc text
  sql: "sql",
  graphql: "graphql",
  gql: "graphql",
  proto: "protobuf",
  diff: "diff",
  patch: "diff",
  nix: "nix",
  sol: "solidity",
};

const FILENAME_TO_LANGUAGE: Record<string, string> = {
  dockerfile: "docker",
  makefile: "makefile",
  gnumakefile: "makefile",
  ".bashrc": "bash",
  ".zshrc": "bash",
  ".profile": "bash",
  ".env": "bash",
  ".gitignore": "bash",
  ".dockerignore": "bash",
};

function getExtension(path: string): string {
  const slashIdx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  const basename = path.slice(slashIdx + 1);
  const dotIdx = basename.lastIndexOf(".");
  if (dotIdx <= 0) return "";
  return basename.slice(dotIdx + 1).toLowerCase();
}

function getBasename(path: string): string {
  const slashIdx = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return path.slice(slashIdx + 1).toLowerCase();
}

/**
 * Resolve a path (and optionally a MIME type) to a Prism grammar name.
 * Returns `null` when no grammar matches — the caller should then render
 * the source as plain text rather than feeding "unknown" to Prism (which
 * still produces an extra wrapper with no highlighting).
 */
export function getPrismLanguageForFile(
  path: string,
  mimeType?: string,
): string | null {
  const basename = getBasename(path);
  const fromBasename = FILENAME_TO_LANGUAGE[basename];
  if (fromBasename) return fromBasename;

  const ext = getExtension(path);
  if (ext && EXTENSION_TO_LANGUAGE[ext]) {
    return EXTENSION_TO_LANGUAGE[ext];
  }

  // MIME-type fallbacks for the handful of cases where the path lacks a
  // useful extension but the server told us what it is.
  if (mimeType) {
    if (mimeType === "text/html") return "markup";
    if (mimeType === "text/css") return "css";
    if (mimeType === "application/json") return "json";
    if (mimeType === "text/markdown") return "markdown";
    if (
      mimeType === "application/javascript" ||
      mimeType === "text/javascript"
    ) {
      return "javascript";
    }
    if (mimeType === "application/x-yaml" || mimeType === "text/yaml") {
      return "yaml";
    }
  }

  return null;
}
