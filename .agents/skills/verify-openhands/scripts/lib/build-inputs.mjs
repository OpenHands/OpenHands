// Repository paths the frontend build reads, relative to the repo root. Their
// tree (plus uncommitted changes) identifies a build: a commit that touches
// none of them reuses the existing build. Anything the app imports from
// outside src/ (config/defaults.json) belongs here too, or a change to it
// would keep serving the old bundle.
export const BUILD_INPUTS = [
  "src",
  "public",
  "package.json",
  "package-lock.json",
  "vite.config.ts",
  "react-router.config.ts",
  "tsconfig.json",
  "tailwind.config.js",
  "config",
];
