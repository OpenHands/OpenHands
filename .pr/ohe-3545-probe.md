# OHE-3545: org hand-off from the cloud's "Back to App"

Real Canvas UI (`react-router dev`), locked to its own origin as on a cloud install, in headless Chromium:

```bash
VITE_LOCK_TO_CLOUD=http://127.0.0.1:5199 VITE_MOCK_API=false npx react-router dev --port 5199 --host 127.0.0.1
```

**Setup**
- Playwright answers the cloud's `/api/**`:
  - `GET /api/organizations` lists "OpenHands-Test" (`org-test`) and the personal workspace (`user-1`);
  - `current_org_id` is `user-1`, as after picking Personal Workspace in the cloud's Settings.
- The tab starts out remembering `{"backendId":"locked-cloud","orgId":"org-test"}`, the org Canvas had before the user went to Settings.

**Steps**
1. Open `/`.
2. Open `/?org=user-1`, the URL the cloud's "Back to App" link now produces.
3. Reload.

## Before (main)

```
[before] 1 Canvas before Settings: address bar query "" | tab selection {"backendId":"locked-cloud","orgId":"org-test"}
[before] 2 Back to App with ?org=user-1: address bar query "?org=user-1" | tab selection {"backendId":"locked-cloud","orgId":"org-test"}
[before] 3 reload: address bar query "?org=user-1" | tab selection {"backendId":"locked-cloud","orgId":"org-test"}
```

The selector still reads "OpenHands Cloud – OpenHands-Test" (`ohe-3545-before.png`).

## After (this branch)

```
[after] 1 Canvas before Settings: address bar query "" | tab selection {"backendId":"locked-cloud","orgId":"org-test"}
[after] 2 Back to App with ?org=user-1: address bar query "" | tab selection {"backendId":"locked-cloud","orgId":"user-1"}
[after] 3 reload: address bar query "" | tab selection {"backendId":"locked-cloud","orgId":"user-1"}
```

The selector reads "OpenHands Cloud – Personal Workspace" (`ohe-3545-after.png`). `?org=` is gone from the address bar, and the choice survives a reload.
