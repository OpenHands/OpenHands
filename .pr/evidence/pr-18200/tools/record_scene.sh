#!/usr/bin/env bash
# Record one before/after scene with control-openhands and log each click.
# Usage: record_scene.sh before|after settings|customize
#   writes $SCRATCH/clips/<run>-<scene>.{mp4,jsonl} (Node >=24 on PATH)
set -euo pipefail
RUN="$1"
SCENE="$2"
# SCRATCH holds verify/<run> (OH_VERIFY_HOME) and receives clips/;
# BEFORE_REPO / AFTER_REPO are the two OpenHands checkouts that were launched.
SCRATCH=${SCRATCH:-$PWD}
REPO=$([ "$RUN" = before ] && echo "${BEFORE_REPO:?}" || echo "${AFTER_REPO:?}")
export PATH=$REPO/.agents/skills/verify-openhands/scripts:$PATH
export OH_VERIFY_RUN=$SCRATCH/verify/$RUN
mkdir -p "$SCRATCH/clips"
LOG="$SCRATCH/clips/$RUN-$SCENE.jsonl"
: >"$LOG"
co() { control-openhands "$@"; }
now() { date +%s%3N; }
PACE=${PACE:-1.6}

# bbox of a selector as "x y w h"
box() { co browser bbox "$1" | python3 -c "import json,sys; b=json.load(sys.stdin)['box']; print(b['x'],b['y'],b['width'],b['height'])"; }

# step <label> <selector> [click args...]: log the target box and click it
step() {
  local label="$1" sel="$2"
  shift 2
  read -r x y w h < <(box "$sel")
  local t0
  t0=$(now)
  co browser click "$sel" "$@" >/dev/null
  local t1
  t1=$(now)
  printf '{"kind":"click","label":"%s","x":%s,"y":%s,"w":%s,"h":%s,"t0":%s,"t1":%s}\n' \
    "$label" "$x" "$y" "$w" "$h" "$t0" "$t1" >>"$LOG"
  sleep "$PACE"
}

co browser viewport tablet >/dev/null
case "$SCENE" in
settings) START=/settings/app; REST='testid=settings-page-subtitle' ;;
customize) START=/mcp; REST='testid=mcp-installed-empty' ;;
esac
co browser goto "$START" >/dev/null
co browser wait "$REST" >/dev/null
co browser hover "$REST" >/dev/null
read -r x y w h < <(box "$REST")
printf '{"kind":"rest","x":%s,"y":%s,"w":%s,"h":%s}\n' "$x" "$y" "$w" "$h" >>"$LOG"
sleep 1.5

co browser record start --feature demo --name "$RUN-$SCENE" >/dev/null
printf '{"kind":"record-start","t":%s}\n' "$(now)" >>"$LOG"
sleep 1.2
printf '{"kind":"status-before-first-click","changes":%s}\n' \
  "$(co browser record status | python3 -c 'import json,sys; print(json.load(sys.stdin)["changes"])')" >>"$LOG"

if [ "$RUN" = before ] && [ "$SCENE" = settings ]; then
  step gear 'testid=backend-selector-settings-link' --expect-url '/settings$'
  step llm 'testid=settings-mobile-hub >> testid=sidebar-settings-/settings/llm' --expect-url '/settings/llm$'
  step gear 'testid=backend-selector-settings-link' --expect-url '/settings$'
  step secrets 'testid=settings-mobile-hub >> testid=sidebar-settings-/settings/secrets' --expect-url '/settings/secrets$'
elif [ "$RUN" = after ] && [ "$SCENE" = settings ]; then
  step open 'testid=settings-compact-navigation >> role=button'
  step llm 'testid=settings-compact-navigation >> testid=sidebar-settings-/settings/llm' --expect-url '/settings/llm$'
  step open 'testid=settings-compact-navigation >> role=button'
  step secrets 'testid=settings-compact-navigation >> testid=sidebar-settings-/settings/secrets' --expect-url '/settings/secrets$'
elif [ "$RUN" = before ] && [ "$SCENE" = customize ]; then
  step customize 'testid=sidebar-skills-link' --expect-url '/customize$'
  step skills 'testid=extensions-mobile-hub >> testid=sidebar-extensions-/skills' --expect-url '/skills$'
else
  step open 'testid=extensions-compact-navigation >> role=button'
  step skills 'testid=extensions-compact-navigation >> testid=sidebar-extensions-/skills' --expect-url '/skills$'
fi
sleep 0.6
co browser record stop | tee "$SCRATCH/clips/$RUN-$SCENE.stop.json" >/dev/null
VIDEO=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('video') or d.get('path'))" "$SCRATCH/clips/$RUN-$SCENE.stop.json")
cp "$VIDEO" "$SCRATCH/clips/$RUN-$SCENE.mp4"
co browser url | grep '"url"'
echo "recorded $RUN-$SCENE -> $SCRATCH/clips/$RUN-$SCENE.mp4"
