import { renderHook } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  __resetActiveStoreForTests,
  NO_BACKEND_ID,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { BACKEND_QUERY_PARAM } from "#/api/backend-registry/url-selection";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";

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

function wrapper({ children }: React.PropsWithChildren) {
  return <ActiveBackendProvider>{children}</ActiveBackendProvider>;
}

const renderScopedPath = () =>
  renderHook(() => useBackendScopedPath(), { wrapper });

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
});

describe("useBackendScopedPath", () => {
  it("pins the active backend onto a conversation path", () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-3" });

    const { result } = renderScopedPath();

    expect(result.current("/conversations/abc")).toBe(
      `/conversations/abc?${BACKEND_QUERY_PARAM}=prod&org=org-3`,
    );
  });

  it("tracks the backend the user switches to", () => {
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: localBackend.id });

    const { result, rerender } = renderScopedPath();
    expect(result.current("/conversations/abc")).toContain(
      `${BACKEND_QUERY_PARAM}=local-1`,
    );

    setActiveSelection({ backendId: cloudBackend.id });
    rerender();

    expect(result.current("/conversations/abc")).toContain(
      `${BACKEND_QUERY_PARAM}=prod`,
    );
  });

  it("leaves the path untouched when no backend is registered", () => {
    setRegisteredBackends([]);

    const { result } = renderScopedPath();

    expect(result.current("/conversations/abc")).toBe("/conversations/abc");
    expect(result.current("/conversations/abc")).not.toContain(NO_BACKEND_ID);
  });
});
