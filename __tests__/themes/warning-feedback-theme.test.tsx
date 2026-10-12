import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AcpConflictWarnings } from "#/components/features/settings/acp-conflict-warnings";
import CanvasExtensionsScreen from "#/routes/canvas-extensions";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import SettingsService from "#/api/settings-service/settings-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";
import { COLOR_THEMES } from "#/themes/color-themes";

vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({
    navigate: vi.fn(),
    currentPath: "/apps",
    conversationId: null,
    isNavigating: false,
  }),
  NavigationProvider: ({ children }: { children: React.ReactNode }) => children,
}));

describe("warning feedback theme roles", () => {
  it("registers --oh-feedback-warning in AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES", () => {
    expect(AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES["--oh-feedback-warning"]).toBe(
      "#f59e0b",
    );
  });

  it("applies text-feedback-warning to ACP conflict warnings", () => {
    render(
      <AcpConflictWarnings
        conflicts={[["test-credential", "conflicting-agent"]]}
      />,
    );
    const warning = screen.getByTestId("acp-credential-conflict-warning");
    expect(warning).toHaveClass("text-feedback-warning");
    expect(warning.className).not.toMatch(/amber-|yellow-/);
  });

  it("applies border-warning/30, bg-warning/5, and text-feedback-warning to Canvas Extensions trust notice", async () => {
    vi.spyOn(CanvasExtensionsService, "listInstalled").mockResolvedValue([]);
    vi.spyOn(SettingsService, "getSettings").mockResolvedValue(
      MOCK_DEFAULT_USER_SETTINGS,
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>
          <CanvasExtensionsScreen />
        </ActiveBackendProvider>
      </QueryClientProvider>,
    );
    // Find the trust notice container
    const trustNotice = await screen.findByText("SETTINGS$APPS_TRUST_NOTICE");
    expect(trustNotice).toHaveClass("border-warning/30");
    expect(trustNotice).toHaveClass("bg-warning/5");
    expect(trustNotice).toHaveClass("text-feedback-warning");
    expect(trustNotice.className).not.toMatch(/amber-|yellow-/);
  });

  it.each(Object.entries(COLOR_THEMES))(
    "meets the 4.5:1 contrast floor in %s across raised, base-secondary, and trust-panel surfaces",
    (_key, theme) => {
      const variables: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      const resolve = (name: string): string => {
        const value =
          variables[name] ?? (name === "--oh-warning" ? "#c9b974" : undefined);
        expect(value, `Missing theme token ${name}`).toBeDefined();
        const reference = /^var\((--[\w-]+)\)$/.exec(value!);
        return reference ? resolve(reference[1]) : value!;
      };
      const hexToRgb = (hex: string): [number, number, number] => [
        Number.parseInt(hex.slice(1, 3), 16) / 255,
        Number.parseInt(hex.slice(3, 5), 16) / 255,
        Number.parseInt(hex.slice(5, 7), 16) / 255,
      ];
      const rgbToLum = (rgb: [number, number, number]): number => {
        const channels = rgb.map((c) =>
          c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4,
        );
        return (
          channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
        );
      };
      const contrast = (l1: number, l2: number) =>
        (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

      const warningTextLum = rgbToLum(
        hexToRgb(resolve("--oh-feedback-warning")),
      );

      // Surface 1: raised surface (Home banner)
      const raisedLum = rgbToLum(hexToRgb(resolve("--oh-surface-raised")));
      expect(contrast(warningTextLum, raisedLum)).toBeGreaterThanOrEqual(4.5);

      // Surface 2: base-secondary (Modals, ACP credentials, onboarding)
      const baseSecondaryLum = rgbToLum(
        hexToRgb(resolve("--oh-color-base-secondary")),
      );
      expect(contrast(warningTextLum, baseSecondaryLum)).toBeGreaterThanOrEqual(
        4.5,
      );

      // Surface 3: Canvas Extensions trust panel (5% warning tint over base)
      const baseRgb = hexToRgb(resolve("--oh-color-base"));
      const warningRgb = hexToRgb(resolve("--oh-warning"));
      const trustPanelRgb: [number, number, number] = [
        warningRgb[0] * 0.05 + baseRgb[0] * 0.95,
        warningRgb[1] * 0.05 + baseRgb[1] * 0.95,
        warningRgb[2] * 0.05 + baseRgb[2] * 0.95,
      ];
      const trustPanelLum = rgbToLum(trustPanelRgb);
      expect(contrast(warningTextLum, trustPanelLum)).toBeGreaterThanOrEqual(
        4.5,
      );
    },
  );
});
