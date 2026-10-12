import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RiskAlert } from "#/components/shared/risk-alert";
import { ErrorMessage } from "#/components/features/home/shared/error-message";
import { SetupFormField } from "#/components/features/manifest/manifest-form-field";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>,
  );
}

describe("Validation and service error theme roles (Issue #18329)", () => {
  describe("RiskAlert", () => {
    it("uses semantic danger border/bg and text-feedback-error instead of raw hex values", () => {
      const { container } = render(
        <RiskAlert
          severity="high"
          title="Security Risk"
          content="Unsafe operation detected"
        />,
      );

      const alert = container.firstChild as HTMLElement;
      expect(alert).toHaveClass("bg-danger/10");
      expect(alert).toHaveClass("border-danger");
      expect(alert).toHaveClass("text-feedback-error");
      expect(alert.className).not.toContain("#4A0709");
      expect(alert.className).not.toContain("#FF0006");
      expect(alert.className).not.toMatch(/text-red-\d+/);
    });
  });

  describe("ErrorMessage", () => {
    it("uses text-feedback-error instead of text-red-500", () => {
      render(
        <ErrorMessage
          isError={true}
          message="Failed to connect"
          testId="test-error"
        />,
      );

      const err = screen.getByTestId("test-error");
      expect(err).toHaveClass("text-feedback-error");
      expect(err.className).not.toMatch(/text-red-\d+/);
    });
  });

  describe("ManifestFormField error styling", () => {
    it("uses border-danger on invalid multiline field and text-feedback-error on error message", () => {
      renderWithClient(
        <SetupFormField
          name="test-multiline"
          field={{
            label: "Test Multiline",
            type: "textarea",
            required: true,
            help: "",
          }}
          value=""
          error="This field is required"
          options={[]}
          repository={null}
          disabled={false}
          onChange={() => {}}
          onBlur={() => {}}
          onRepositoryChange={() => {}}
        />,
      );

      const textarea = screen.getByTestId("setup-field-test-multiline");
      expect(textarea).toHaveClass("border-danger");
      expect(textarea.className).not.toMatch(/border-red-\d+/);

      const errorText = screen.getByTestId("setup-field-test-multiline-error");
      expect(errorText).toHaveClass("text-feedback-error");
      expect(errorText.className).not.toMatch(/text-red-\d+/);
    });

    it("uses border-danger on invalid tarball-upload field", () => {
      renderWithClient(
        <SetupFormField
          name="test-tarball"
          field={{
            label: "Test Tarball",
            type: "tarball-upload",
            required: true,
            help: "",
          }}
          value=""
          error="Upload failed"
          options={[]}
          repository={null}
          disabled={false}
          onChange={() => {}}
          onBlur={() => {}}
          onRepositoryChange={() => {}}
        />,
      );

      const fileInput = screen.getByTestId("setup-field-test-tarball");
      expect(fileInput).toHaveClass("border-danger");
      expect(fileInput.className).not.toMatch(/border-red-\d+/);

      const errorText = screen.getByTestId("setup-field-test-tarball-error");
      expect(errorText).toHaveClass("text-feedback-error");
      expect(errorText.className).not.toMatch(/text-red-\d+/);
    });
  });
});
