#!/bin/bash
# Evidence for the bump-pins pull request: replay the last upstream bump with
# scripts/bump-pins.mjs and compare with what was merged by hand.
#
# Takes a checkout from just before #19658 (agent-server 1.61.0, automation
# 1.31.0, extensions 0.41.0), runs the script with the versions that #19658 and
# #19659 moved the pins to, and diffs the seven files against main after #19659.
# Local only: it clones into a temporary directory and pushes nothing.
#
# Usage: bash .pr/bump-pins/replay.sh        (from the repository root)
set -euo pipefail
script="$PWD/scripts/bump-pins.mjs"
work="$(mktemp -d)/OpenHands"
git clone --quiet --filter=blob:none --no-checkout https://github.com/OpenHands/OpenHands.git "$work"
cd "$work"
first="$(git log --format=%H -1 --grep='(#19658)' origin/main)"
second="$(git log --format=%H -1 --grep='(#19659)' origin/main)"
git checkout --quiet "$first~1"
cp "$script" scripts/bump-pins.mjs

pins='"agent-server " + c.versions.agentServer + ", automation " + c.versions.automation + ", extensions " + p.dependencies["@openhands/extensions"] + ", client " + p.dependencies["@openhands/typescript-client"]'
show() { node -p "const c = require('./config/defaults.json'), p = require('./package.json'); $pins"; }

echo "\$ git checkout $(git rev-parse --short "$first")~1    # just before #19658"
echo "pins: $(show)"
echo
echo "\$ PIN_SDK=1.62.0 PIN_AUTOMATION=1.32.1 PIN_EXTENSIONS=0.42.0 node scripts/bump-pins.mjs"
PIN_SDK=1.62.0 PIN_AUTOMATION=1.32.1 PIN_EXTENSIONS=0.42.0 node scripts/bump-pins.mjs | tr ',' '\n' | sed 's/^ /         /'
echo "pins: $(show)"
echo
echo "\$ git diff --stat"
git diff --stat -- . ':!scripts/bump-pins.mjs'
echo
files=(config/defaults.json package.json package-lock.json .agents/skills/local-stack-runtime/references/guide.md scripts/dev-safe.mjs scripts/check-sdk-version-sync.mjs __tests__/scripts/dev-safe.test.ts)
echo "\$ git diff --quiet $(git rev-parse --short "$second") -- <the seven files>    # main after #19658 + #19659"
if git diff --quiet "$second" -- "${files[@]}"; then
  echo "identical to what the two pull requests merged"
else
  echo "DIFFERENT from what was merged:"
  git diff --stat "$second" -- "${files[@]}"
fi
echo
echo "\$ PIN_SDK=1.62.0 PIN_AUTOMATION=1.32.1 PIN_EXTENSIONS=0.42.0 node scripts/bump-pins.mjs    # again"
PIN_SDK=1.62.0 PIN_AUTOMATION=1.32.1 PIN_EXTENSIONS=0.42.0 node scripts/bump-pins.mjs
