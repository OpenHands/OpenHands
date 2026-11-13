import { describe, expect, it } from "vitest";
import { LOCKED_CLOUD_BACKEND_ID } from "#/api/backend-registry/default-backend";
import {
  BACKEND_QUERY_PARAM,
  ORG_QUERY_PARAM,
  readBackendSelectionFromUrl,
  readLockedCloudOrgFromUrl,
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

  it("keeps a fragment after existing query parameters intact and after the query", () => {
    const path = withBackendSelectionParams(
      "/conversations/abc?tab=files#detail",
      {
        backend: localBackend,
        orgId: null,
      },
    );

    expect(path).toBe(
      `/conversations/abc?tab=files&${BACKEND_QUERY_PARAM}=local-1#detail`,
    );
  });

  it("keeps a fragment on a path without query parameters after the query", () => {
    const path = withBackendSelectionParams("/conversations/abc#detail", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(
      `/conversations/abc?${BACKEND_QUERY_PARAM}=local-1#detail`,
    );
  });

  it("does not treat a ? inside the fragment as a query separator", () => {
    const path = withBackendSelectionParams("/conversations/abc#detail?x=1", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(
      `/conversations/abc?${BACKEND_QUERY_PARAM}=local-1#detail?x=1`,
    );
  });

  it("round-trips an empty fragment verbatim", () => {
    const path = withBackendSelectionParams("/conversations/abc#", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(`/conversations/abc?${BACKEND_QUERY_PARAM}=local-1#`);
  });

  it("keeps the org id and the fragment together", () => {
    const path = withBackendSelectionParams("/conversations/abc#detail", {
      backend: cloudBackend,
      orgId: "org-7",
    });

    expect(path).toBe(
      `/conversations/abc?${BACKEND_QUERY_PARAM}=prod&${ORG_QUERY_PARAM}=org-7#detail`,
    );
  });

  it("keeps query data that itself contains a ?", () => {
    const path = withBackendSelectionParams("/conversations/abc?next=/a?b=1", {
      backend: localBackend,
      orgId: null,
    });

    expect(path).toBe(
      `/conversations/abc?next=%2Fa%3Fb%3D1&${BACKEND_QUERY_PARAM}=local-1`,
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

describe("readLockedCloudOrgFromUrl", () => {
  const lockedCloud: Backend = {
    id: LOCKED_CLOUD_BACKEND_ID,
    name: "OpenHands Cloud",
    host: "https://app.all-hands.dev",
    apiKey: "",
    kind: "cloud",
    authMode: "cookie",
  };

  it("reads the org the cloud's web app hands back", () => {
    expect(
      readLockedCloudOrgFromUrl([lockedCloud], `?${ORG_QUERY_PARAM}=org-7`),
    ).toEqual({ backendId: LOCKED_CLOUD_BACKEND_ID, orgId: "org-7" });
  });

  it("leaves a URL that also names a backend to readBackendSelectionFromUrl", () => {
    expect(
      readLockedCloudOrgFromUrl(
        [lockedCloud],
        `?${BACKEND_QUERY_PARAM}=${LOCKED_CLOUD_BACKEND_ID}&${ORG_QUERY_PARAM}=org-7`,
      ),
    ).toBeNull();
  });

  it("ignores the org when Canvas is not locked to a cloud host", () => {
    expect(
      readLockedCloudOrgFromUrl(
        [localBackend, cloudBackend],
        `?${ORG_QUERY_PARAM}=org-7`,
      ),
    ).toBeNull();
  });

  it("ignores an empty or absent org", () => {
    expect(readLockedCloudOrgFromUrl([lockedCloud], "")).toBeNull();
    expect(readLockedCloudOrgFromUrl([lockedCloud], "?tab=files")).toBeNull();
    expect(
      readLockedCloudOrgFromUrl([lockedCloud], `?${ORG_QUERY_PARAM}=`),
    ).toBeNull();
  });
});
