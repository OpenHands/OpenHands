# Dependency audits

The `Audit` workflow (`.github/workflows/audit.yml`) fails on every **high** or
**critical** npm advisory. It runs on pull requests, on pushes to `main`, and on
a weekly schedule so advisories published against untouched dependencies still
surface. Weekly Dependabot npm PRs (`.github/dependabot.yml`) land most of the
patch bumps that clear them.

Reproduce a CI failure locally:

```bash
npm audit --audit-level=high --package-lock-only
```

Then fix it in the usual order of preference:

1. `npm audit fix --package-lock-only` for transitive patch bumps.
2. Bump the direct dependency in `package.json` when the fix is out of its
   pinned range.
3. Pin the transitive package via the `overrides` block in `package.json` when
   the parent hasn't released a fix yet.

## Exception process

Only when none of the above works — no fixed version exists, or the fix requires
a breaking upgrade we can't take right now — add an allowlist entry to
`.github/audit-ci.jsonc`:

```jsonc
"allowlist": [
  // GHSA-xxxx-xxxx-xxxx: <package> — dev-only (build tooling, never shipped to
  // users). No fixed release yet; revisit after <upstream issue/date>.
  "GHSA-xxxx-xxxx-xxxx"
]
```

Rules for an exception:

- It must be justified in the PR description **and** in a comment next to the
  entry: why the advisory is not exploitable here (dev-only dependency, code
  path unreachable, etc.) and what would let us drop the entry.
- Scope it as narrowly as possible: prefer the advisory ID (or
  `GHSA-xxxx-xxxx-xxxx|parent>child` for a single path) over a bare package
  name, so the gate keeps firing for other packages and other advisories.
- Entries are temporary. `audit-ci` reports allowlisted advisories that are no
  longer found, which is the signal to delete the entry.
