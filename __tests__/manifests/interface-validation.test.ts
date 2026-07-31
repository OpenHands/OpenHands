import { describe, expect, it } from "vitest";
import { validateInterfaceManifest } from "#/manifests/interface-validation";
import type { InterfaceValidationContext } from "#/manifests/interface-validation";
import {
  createInterfaceManifest,
  createInterfaceManifestWith,
} from "./manifest-test-data";

const CONTEXT: InterfaceValidationContext = {
  catalogIds: new Set([
    "github-pr-reviewer",
    "github-repo-monitor",
    "slack-channel-monitor",
  ]),
  mountedRoutes: {
    list: "/automations",
    setup: "/automations/new/:automationId",
    detail: "/automations/:automationId",
  },
};

describe("validateInterfaceManifest", () => {
  it("admits a well-formed manifest", () => {
    // Arrange
    const manifest = createInterfaceManifest();

    // Act
    const result = validateInterfaceManifest(manifest, CONTEXT);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });

  // Each case is a separate invariant the host enforces on data authored in
  // another repository. A manifest that trips any of them reverts the whole
  // interface to the host's defaults.
  it.each([
    [
      "a version this host cannot interpret",
      { version: "2.0" },
    ],
    [
      // The host serves what it has registrations for; a manifest cannot remap
      // the router table, only own link construction against it.
      "a route the host has no registration for",
      {
        routes: {
          list: "/automations",
          setup: "/workflows/new/:automationId",
          detail: "/automations/:automationId",
        },
      },
    ],
    [
      "markup inside user-visible copy",
      {
        navigation: {
          sidebar: { label: "<img src=x onerror=alert(1)>" },
          commandMenu: {
            title: "Automations",
            description: "Review them.",
            keywords: "automate",
          },
        },
      },
    ],
    [
      "a documentation link outside the product documentation",
      { docsUrl: "https://evil.example/phishing" },
    ],
    [
      "an endpoint that is not a rooted service-relative path",
      {
        endpoints: {
          ...createInterfaceManifest().endpoints,
          list: "https://evil.example/v1",
        },
      },
    ],
    [
      "an id endpoint without its {id} substitution",
      {
        endpoints: {
          ...createInterfaceManifest().endpoints,
          detail: "/v1/latest",
        },
      },
    ],
    [
      "a substitution on an endpoint the host calls without an id",
      {
        endpoints: {
          ...createInterfaceManifest().endpoints,
          list: "/v1/{id}",
        },
      },
    ],
    [
      "a featured automation the catalog does not publish",
      { featuredAutomationIds: ["github-pr-reviewer", "unpublished-entry"] },
    ],
    [
      "an edit field for a property the host cannot edit",
      {
        edit: {
          title: "Edit automation",
          fields: {
            tarballPath: { type: "text", label: "Tarball", required: false },
          },
        },
      },
    ],
    [
      "constraints on a field that is not a number",
      {
        edit: {
          title: "Edit automation",
          fields: {
            name: {
              type: "text",
              label: "Name",
              required: true,
              constraints: { max: 50 },
            },
          },
        },
      },
    ],
    [
      "a key this host does not read",
      { dashboards: [] },
    ],
  ])("refuses %s", (_case, overrides) => {
    // Arrange
    const candidate = createInterfaceManifestWith(overrides);

    // Act
    const result = validateInterfaceManifest(candidate, CONTEXT);

    // Assert
    expect(result.valid).toBe(false);
  });

  it("reports every problem at once so an author sees the whole picture", () => {
    // Arrange
    const candidate = createInterfaceManifestWith({
      docsUrl: "https://evil.example/",
      featuredAutomationIds: ["unpublished-entry"],
    });

    // Act
    const { errors } = validateInterfaceManifest(candidate, CONTEXT);

    // Assert
    expect(errors).toHaveLength(2);
  });
});
