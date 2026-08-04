import { describe, expect, it } from "vitest";
import {
  BACKEND_QUERY_PARAM,
  ORG_QUERY_PARAM,
  readBackendSelectionFromUrl,
  withBackendSelectionParams,
} from "#/api/backend-registry/url-selection";
import type { Backend } from "#/api/backend-registry/types";

const localBackend: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:9000",
  apiKey: "k",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

const backends = [localBackend, cloudBackend];

describe("withBackendSelectionParams", () => {
  it("pins the active backend id onto the path", () => {
    const path = withBackendSelectionParams("/conversations/abc", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(`/conversations/abc?${BACKEND_QUERY_PARAM}=local-1`);
  });

  it("includes the org id for a cloud backend", () => {
    const path = withBackendSelectionParams("/conversations/abc", {
      backend: cloudBackend,
      orgId: "org-7",
    });

    expect(path).toBe(
      `/conversations/abc?${BACKEND_QUERY_PARAM}=prod&${ORG_QUERY_PARAM}=org-7`,
    );
  });

  it("omits the org id when there is none", () => {
    const path = withBackendSelectionParams("/conversations/abc", {
      backend: cloudBackend,
      orgId: null,
    });

    expect(path).not.toContain(ORG_QUERY_PARAM);
  });

  it("preserves query parameters already on the path", () => {
    const path = withBackendSelectionParams("/conversations/abc?tab=files", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(
      `/conversations/abc?tab=files&${BACKEND_QUERY_PARAM}=local-1`,
    );
  });
});

describe("readBackendSelectionFromUrl", () => {
  it("reads a registered backend id", () => {
    expect(
      readBackendSelectionFromUrl(backends, `?${BACKEND_QUERY_PARAM}=local-1`),
    ).toEqual({ backendId: "local-1", orgId: null });
  });

  it("reads the org id alongside the backend id", () => {
    expect(
      readBackendSelectionFromUrl(
        backends,
        `?${BACKEND_QUERY_PARAM}=prod&${ORG_QUERY_PARAM}=org-7`,
      ),
    ).toEqual({ backendId: "prod", orgId: "org-7" });
  });

  it("ignores a backend id that is not registered", () => {
    expect(
      readBackendSelectionFromUrl(backends, `?${BACKEND_QUERY_PARAM}=gone`),
    ).toBeNull();
  });

  it("ignores an empty or absent parameter", () => {
    expect(readBackendSelectionFromUrl(backends, "")).toBeNull();
    expect(readBackendSelectionFromUrl(backends, "?tab=files")).toBeNull();
    expect(
      readBackendSelectionFromUrl(backends, `?${BACKEND_QUERY_PARAM}=`),
    ).toBeNull();
  });
});
