import { describe, expect, it } from "vitest";
import { validateManifest } from "#/manifests/manifest-validation";
import { createManifest, createManifestWith } from "./manifest-test-data";

describe("validateManifest", () => {
  it("admits a well-formed manifest", () => {
    // Arrange
    const manifest = createManifest();

    // Act
    const result = validateManifest(manifest);

    // Assert
    expect(result).toEqual({ valid: true, errors: [] });
  });

  // Each case is a separate invariant the host enforces on data authored in
  // another repository. A manifest that trips any of them must not render.
  it.each([
    [
      "a manifest version this host cannot interpret",
      { manifestVersion: "2.0" },
    ],
    [
      "an action outside the allowlist",
      {
        submit: {
          action: "shell.exec",
          endpoint: { method: "POST", path: "/v1/preset/prompt" },
          payload: { name: "x" },
          onSuccess: { behavior: "navigate", to: "/widgets/1" },
          onError: { behavior: "stayOnForm", errorTarget: "field" },
        },
      },
    ],
    [
      "a request path outside the service namespace",
      {
        submit: {
          action: "automation.create",
          endpoint: { method: "POST", path: "/internal/admin" },
          payload: { name: "x" },
          onSuccess: { behavior: "navigate", to: "/widgets/1" },
          onError: { behavior: "stayOnForm", errorTarget: "field" },
        },
      },
    ],
    [
      "a redirect that leaves the application",
      {
        submit: {
          action: "automation.create",
          endpoint: { method: "POST", path: "/v1/preset/prompt" },
          payload: { name: "x" },
          onSuccess: { behavior: "navigate", to: "https://example.com/steal" },
          onError: { behavior: "stayOnForm", errorTarget: "field" },
        },
      },
    ],
    [
      "markup inside user-visible copy",
      { description: "<img src=x onerror=alert(1)>" },
    ],
    [
      "a placeholder namespace the host does not expose",
      {
        review: {
          title: "Review",
          summary: [{ label: "Token", value: "{{secrets.githubToken}}" }],
          confirmLabel: "Create",
        },
      },
    ],
    [
      "a credential requirement carrying anything beyond its name",
      {
        requires: {
          integrations: [],
          secrets: [
            {
              key: "API_TOKEN",
              label: "API token",
              help: "Needed to call the service.",
              required: true,
              value: "ghp_realtokenvalue",
            },
          ],
          onUnmet: { behavior: "block", message: "Provide a token." },
        },
      },
    ],
  ])("refuses %s", (_case, overrides) => {
    // Arrange
    const candidate = createManifestWith(overrides);

    // Act
    const result = validateManifest(candidate);

    // Assert
    expect(result.valid).toBe(false);
  });

  it("reports every problem at once so an author sees the whole picture", () => {
    // Arrange
    const candidate = createManifestWith({ name: "", category: "" });

    // Act
    const { errors } = validateManifest(candidate);

    // Assert
    expect(errors).toHaveLength(2);
  });
});
