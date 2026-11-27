#!/usr/bin/env bash
# Custom Vercel install command.
#
# npm normalizes any GitHub URL it finds in package.json (including
# `git+https://github.com/...` and the `github:owner/repo` shorthand) to
# `git+ssh://git@github.com/...` when it writes package-lock.json. Vercel's
# build environment has no SSH key for GitHub, so npm cannot clone the
# `@openhands/typescript-client` git dependency and silently falls back to a
# stale cached copy — producing the dreaded
# `[MISSING_EXPORT] ConversationClient is not exported by
# node_modules/@openhands/typescript-client/dist/clients.js` at bundle time.
#
# Two defensive measures here:
#   1. Rewrite any `git+ssh://git@github.com/` URLs in package-lock.json
#      to `git+https://github.com/` before invoking npm so the lockfile
#      Vercel actually consumes is HTTPS-only, regardless of which lockfile
#      shape happened to be committed.
#   2. Configure git globally to translate the matching ssh forms into
#      https — this catches anything npm has already cached as an ssh URL
#      and any future git deps that hit the same bug.
#
# See https://github.com/OpenHands/agent-canvas/issues/384 for the original
# bug report.
set -euo pipefail

if [ -f package-lock.json ]; then
  sed -i 's|git+ssh://git@github.com/|git+https://github.com/|g' package-lock.json
fi

git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"
git config --global url."https://github.com/".insteadOf "git@github.com:"

npm ci
