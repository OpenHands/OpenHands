import posthog from "posthog-js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { handleCaptureConsent } from "#/utils/handle-capture-consent";

describe("handleCaptureConsent", () => {
  const optInSpy = vi.spyOn(posthog, "opt_in_capturing");
  const optOutSpy = vi.spyOn(posthog, "opt_out_capturing");
  const hasOptedInSpy = vi.spyOn(posthog, "has_opted_in_capturing");
  const hasOptedOutSpy = vi.spyOn(posthog, "has_opted_out_capturing");

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should opt out of of capturing", () => {
    handleCaptureConsent(posthog, false);

    expect(optOutSpy).toHaveBeenCalled();
    expect(optInSpy).not.toHaveBeenCalled();
  });

  it("should opt in to capturing if the user consents", () => {
    handleCaptureConsent(posthog, true);

    expect(optInSpy).toHaveBeenCalled();
    expect(optOutSpy).not.toHaveBeenCalled();
  });

  it("should not opt in to capturing if the user is already opted in", () => {
    hasOptedInSpy.mockReturnValueOnce(true);
    handleCaptureConsent(posthog, true);

    expect(optInSpy).not.toHaveBeenCalled();
    expect(optOutSpy).not.toHaveBeenCalled();
  });

  it("should not opt out of capturing if the user is already opted out", () => {
    hasOptedOutSpy.mockReturnValueOnce(true);
    handleCaptureConsent(posthog, false);

    expect(optOutSpy).not.toHaveBeenCalled();
    expect(optInSpy).not.toHaveBeenCalled();
  });
});
