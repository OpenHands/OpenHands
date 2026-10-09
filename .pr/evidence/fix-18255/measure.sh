#!/usr/bin/env bash
# Probe one verify-openhands run for #18255: main's height and scroller,
# selector clipping, pinned desktop nav, phone hub/Back, scroll carry-over.
# Usage: measure.sh <run-id> <repo>   (prints one JSON object per line)
set -uo pipefail
RUN="$1"
REPO="$2"
# OH_VERIFY_HOME holds the runs (<run-id> directories); Node >=24 must be on PATH.
S=${OH_VERIFY_HOME:?}
export PATH=$REPO/.agents/skills/verify-openhands/scripts:$PATH
export OH_VERIFY_RUN=$S/$RUN
co() { control-openhands "$@"; }
ev() { co browser eval "$1" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("value")))'; }
emit() { printf '{"run":"%s","check":"%s","viewport":"%s","route":"%s","value":%s}\n' "$RUN" "$1" "$2" "$3" "$4"; }

MAIN='(() => { const m = document.querySelector("main"); const r = m.getBoundingClientRect(); const outer = m.closest("[class*=overflow-auto]:not(main)"); return { mainBottom: Math.round(r.bottom), viewport: innerHeight, mainScrolls: m.scrollHeight > m.clientHeight + 1, outerScrolls: [...document.querySelectorAll("div")].some(d => d !== m && d.contains(m) && /auto|scroll/.test(getComputedStyle(d).overflowY) && d.scrollHeight > d.clientHeight + 1) }; })()'
LIST='(() => { const n = document.querySelector("[data-testid=settings-compact-navigation] nav"); if (!n) return null; const nb = n.getBoundingClientRect().bottom; const mb = document.querySelector("main").getBoundingClientRect().bottom; return { listBottom: Math.round(nb), mainBottom: Math.round(mb), clipped: mb < nb - 1 }; })()'
SCROLLTOP='(() => { const m = document.querySelector("main"); const outer = [...document.querySelectorAll("div")].find(d => d.contains(m) && /auto|scroll/.test(getComputedStyle(d).overflowY) && d.scrollHeight > d.clientHeight + 1); return { main: Math.round(m.scrollTop), outer: outer ? Math.round(outer.scrollTop) : 0 }; })()'

for vp in desktop tablet phone; do
  co browser viewport $vp >/dev/null
  for route in /settings/secrets /settings/app; do
    co browser goto $route >/dev/null
    co browser wait 'testid=settings-screen' >/dev/null
    sleep 0.6
    emit layout $vp $route "$(ev "$MAIN")"
  done
done

# Tablet selector (#18200 only): open it on the short pages.
co browser viewport tablet >/dev/null
for route in /settings/secrets /settings/agents; do
  co browser goto $route >/dev/null
  sleep 0.6
  if [ "$(co browser count 'testid=settings-compact-navigation' | python3 -c 'import json,sys; print(json.load(sys.stdin).get("count", 0))')" != 0 ]; then
    co browser click 'testid=settings-compact-navigation >> role=button' >/dev/null
    emit selector tablet $route "$(ev "$LIST")"
    name=$(echo "$route" | tr / -)
    co browser screenshot --feature F09.tablet-selector --name "$RUN$name" >/dev/null
    co browser press Escape >/dev/null
  fi
done

# Scroll position after leaving and re-entering a long page at tablet
# (gear -> hub -> Application); only Application is long in fresh state.
co browser goto /settings/app >/dev/null
sleep 0.6
co browser scroll 'testid=settings-page-subtitle' --by 400 >/dev/null
emit scroll-before-switch tablet /settings/app "$(ev "$SCROLLTOP")"
co browser click 'testid=backend-selector-settings-link' --expect-url '/settings$' >/dev/null
co browser click 'testid=settings-mobile-hub >> testid=sidebar-settings-/settings/app' --expect-url '/settings/app$' >/dev/null
sleep 0.6
emit scroll-after-switch tablet /settings/app "$(ev "$SCROLLTOP")"

# Desktop: the Settings nav stays pinned while a long page scrolls.
co browser viewport desktop >/dev/null
co browser goto /settings/app >/dev/null
sleep 0.6
y0=$(co browser bbox 'testid=sidebar-settings-/settings/agents' | python3 -c 'import json,sys; print(json.load(sys.stdin)["box"]["y"])')
co browser scroll 'testid=settings-page-subtitle' --by 600 >/dev/null
y1=$(co browser bbox 'testid=sidebar-settings-/settings/agents' | python3 -c 'import json,sys; print(json.load(sys.stdin)["box"]["y"])')
emit desktop-nav-pinned desktop /settings/app "{\"navYBefore\":$y0,\"navYAfterScroll600\":$y1,\"scroll\":$(ev "$SCROLLTOP")}"
co browser goto /settings/app >/dev/null
sleep 0.8
co browser screenshot --feature F09.height --name "$RUN-desktop-app" >/dev/null

# Phone: hub -> Secrets -> Back returns to the hub.
co browser viewport phone >/dev/null
co browser goto /settings >/dev/null
co browser click 'testid=settings-mobile-hub >> testid=sidebar-settings-/settings/secrets' --expect-url '/settings/secrets$' >/dev/null
co browser click 'testid=sidebar-mobile-back-button' --expect-url '/settings$' >/dev/null
emit phone-hub-back phone /settings "$(co browser visible 'testid=settings-mobile-hub' | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin).get("visible")))')"
co browser errors --app-only | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps({"run":"'"$RUN"'","check":"app-errors","count":len(d.get("errors",[]))}))'
