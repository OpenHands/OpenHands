// @vitest-environment node
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

/**
 * Where the feedback control is mounted is load-bearing, and nothing else
 * catches it.
 *
 * `useOptionalConversationId()` reads `NavigationContext`, which is created with
 * a *default value* (`conversationId: null`). Rendering outside the provider
 * therefore does not throw — it silently yields no conversation id, which is
 * exactly what happened when the control was mounted in `root.tsx` as a sibling
 * of `<Outlet />`. Every feedback event carried `conversation_id: undefined` and
 * the component tests could not see it, because they stub the hook.
 *
 * `ReactRouterNavigationProvider` is mounted inside the route tree, in
 * `root-layout.tsx`, so the control has to live there too. These assertions read
 * the source rather than render it: the failure mode is a mount location, not a
 * behaviour a unit test can observe once the hook is stubbed.
 */

const read = (relativePath: string) =>
  readFileSync(join(process.cwd(), relativePath), "utf8");

describe("FeedbackLauncher mount position", () => {
  it("is mounted from root-layout, which is inside the navigation provider", () => {
    const rootLayout = read("src/routes/root-layout.tsx");

    expect(rootLayout).toContain("<FeedbackLauncher />");
    expect(rootLayout).toContain("ReactRouterNavigationProvider");

    // The launcher must sit inside the provider, not merely in the same file.
    const providerOpens = rootLayout.indexOf("<ReactRouterNavigationProvider>");
    const providerCloses = rootLayout.indexOf(
      "</ReactRouterNavigationProvider>",
    );
    const launcher = rootLayout.indexOf("<FeedbackLauncher />");

    expect(providerOpens).toBeGreaterThan(-1);
    expect(launcher).toBeGreaterThan(providerOpens);
    expect(launcher).toBeLessThan(providerCloses);
  });

  it("is not mounted from root, which renders outside that provider", () => {
    expect(read("src/root.tsx")).not.toContain("FeedbackLauncher");
  });

  it("still relies on the context default that made the old mount fail quietly", () => {
    // If NavigationContext ever starts throwing outside its provider, this
    // guard is redundant and can go.
    expect(read("src/context/navigation-context.tsx")).toContain(
      "conversationId: null",
    );
  });
});
