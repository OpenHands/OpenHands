import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockUseConfig = vi.fn();
const mockUseIsAuthed = vi.fn();
const mockGetLoginMethod = vi.fn();

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => mockUseConfig(),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => mockUseIsAuthed(),
}));

vi.mock("#/utils/local-storage", () => ({
  LoginMethod: {
    GITHUB: "github",
    GITLAB: "gitlab",
    BITBUCKET: "bitbucket",
    BITBUCKET_DATA_CENTER: "bitbucket_data_center",
    AZURE_DEVOPS: "azure_devops",
    ENTERPRISE_SSO: "enterprise_sso",
  },
  getLoginMethod: () => mockGetLoginMethod(),
}));


vi.mock("#/hooks/use-is-on-intermediate-page", () => ({
  useIsOnIntermediatePage: () => true,
}));

import { useAutoLogin } from "#/hooks/use-auto-login";

describe("useAutoLogin", () => {
  const acceptTosUrl =
    "https://ohpr-13306-497.staging.all-hands.dev/accept-tos?redirect_url=" +
    encodeURIComponent(
      "https://ohpr-13306-497.auth.staging.all-hands.dev/realms/allhands/protocol/openid-connect/auth?client_id=allhands&response_type=code",
    );

  beforeEach(() => {
    vi.stubGlobal("location", { href: acceptTosUrl });

    mockUseConfig.mockReturnValue({
      data: {
        app_mode: "saas",
        auth_url: "ohpr-13306-497.auth.staging.all-hands.dev",
      },
      isLoading: false,
    });

    mockUseIsAuthed.mockReturnValue({
      data: false,
      isLoading: false,
    });

    mockGetLoginMethod.mockReturnValue("github");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("should not restart OAuth while the user is already on /accept-tos", async () => {
    renderHook(() => useAutoLogin());

    await waitFor(() => {
      expect(window.location.href).toBe(acceptTosUrl);
    });
  });
});
