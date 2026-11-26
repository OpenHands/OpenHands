import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ProviderUploadLimitNote } from "#/components/features/settings/llm-settings/provider-upload-limit-note";

// Override the global react-i18next mock for this file so the note's
// interpolated values are substituted the way the production runtime does.
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  const TEMPLATES: Record<string, string> = {
    SETTINGS$PROVIDER_UPLOAD_LIMIT:
      "{{provider}} accepts images up to {{limitMb}}MB. Other attachments are limited to {{attachmentLimitMb}}MB.",
    SETTINGS$PROVIDER_UPLOAD_LIMIT_UNKNOWN:
      "No published image limit is known for {{provider}}, so images are limited to {{limitMb}}MB.",
  };
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, opts?: Record<string, unknown>) => {
        const tpl = TEMPLATES[key] ?? key;
        if (!opts) return tpl;
        return tpl.replace(/\{\{(\w+)\}\}/g, (_match, name) =>
          name in opts ? String(opts[name]) : "",
        );
      },
      i18n: { language: "en", exists: () => false },
    }),
  };
});

describe("ProviderUploadLimitNote", () => {
  it("shows the provider's own image limit", () => {
    render(<ProviderUploadLimitNote provider="anthropic" />);

    expect(
      screen.getByTestId("llm-provider-upload-limit-note").textContent,
    ).toBe(
      "Anthropic accepts images up to 5MB. Other attachments are limited to 3MB.",
    );
  });

  it("shows a provider with a larger limit", () => {
    render(<ProviderUploadLimitNote provider="gemini" />);

    expect(
      screen.getByTestId("llm-provider-upload-limit-note").textContent,
    ).toContain("20MB");
  });

  it("says so when the provider has no published limit", () => {
    render(<ProviderUploadLimitNote provider="ollama" />);

    const text = screen.getByTestId(
      "llm-provider-upload-limit-note",
    ).textContent;
    expect(text).toContain("No published image limit is known for Ollama");
    expect(text).toContain("3MB");
  });

  it("renders nothing when no provider is selected", () => {
    render(<ProviderUploadLimitNote provider={null} />);

    expect(screen.queryByTestId("llm-provider-upload-limit-note")).toBeNull();
  });
});
