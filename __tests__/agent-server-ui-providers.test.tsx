import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, useQueryClient } from "@tanstack/react-query";
import { createInstance } from "i18next";
import { initReactI18next, useTranslation } from "react-i18next";

vi.mock("react-i18next", async (importOriginal) =>
  importOriginal<typeof import("react-i18next")>(),
);

import {
  AGENT_SERVER_UI_SCOPE_SELECTOR,
  AgentServerUIRoot,
  AgentServerUIProviders,
  OPENHANDS_I18N_NAMESPACE,
  getDefaultI18n,
  getDefaultQueryClient,
  getI18n,
  getQueryClient,
  queryClient,
  setI18n,
  setQueryClient,
} from "#/index";
import i18n from "#/i18n";

const telemetryProviderMock = vi.hoisted(() => vi.fn());
vi.mock("#/components/providers/telemetry-provider", () => ({
  TelemetryProvider: (props: {
    children: React.ReactNode;
    config?: unknown;
  }) => {
    telemetryProviderMock(props);
    return props.children;
  },
}));

const BaseProbe = ({ translation }: { translation?: string }) => {
  const currentQueryClient = useQueryClient();

  return (
    <div>
      <div data-testid="query-client-kind">
        {currentQueryClient === getDefaultQueryClient() ? "default" : "custom"}
      </div>
      <div data-testid="query-client-value">
        {String(queryClient.getQueryData(["provider-probe"]))}
      </div>
      {translation && <div data-testid="translation-value">{translation}</div>}
      <div data-testid="imperative-translation-value">
        {i18n.t("PROVIDER$LABEL")}
      </div>
    </div>
  );
};

const DefaultProbe = () => <BaseProbe />;

const CustomProbe = () => {
  const { t } = useTranslation(OPENHANDS_I18N_NAMESPACE);

  return <BaseProbe translation={t("PROVIDER$LABEL")} />;
};

const createTestI18n = async (value: string) => {
  const instance = createInstance();

  await instance.use(initReactI18next).init({
    lng: "en",
    fallbackLng: "en",
    ns: ["host", OPENHANDS_I18N_NAMESPACE],
    defaultNS: "host",
    interpolation: { escapeValue: false },
    resources: {
      en: {
        host: {
          PROVIDER$LABEL: "Host provider",
        },
        [OPENHANDS_I18N_NAMESPACE]: {
          PROVIDER$LABEL: value,
        },
      },
    },
  });

  return instance;
};

afterEach(() => {
  cleanup();
  getDefaultQueryClient().removeQueries({ queryKey: ["provider-probe"] });
  setQueryClient();
  setI18n();
  vi.restoreAllMocks();
});

describe("AgentServerUIProviders", () => {
  it("exports and uses the default query client and i18n instances when props are omitted", async () => {
    const defaultI18n = getDefaultI18n();

    defaultI18n.addResourceBundle(
      "en",
      OPENHANDS_I18N_NAMESPACE,
      { PROVIDER$LABEL: "Default provider" },
      true,
      true,
    );
    await defaultI18n.changeLanguage("en");

    getDefaultQueryClient().setQueryData(["provider-probe"], "default-client");

    render(
      <AgentServerUIProviders>
        <DefaultProbe />
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("query-client-kind")).toHaveTextContent(
      "default",
    );
    expect(screen.getByTestId("query-client-value")).toHaveTextContent(
      "default-client",
    );

    await waitFor(() => {
      expect(
        screen.getByTestId("imperative-translation-value"),
      ).toHaveTextContent("Default provider");
    });

    expect(getQueryClient()).toBe(getDefaultQueryClient());
  });

  it("injects a custom query client and i18n instance without conflicting with imperative callers", async () => {
    const customQueryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    const customI18n = await createTestI18n("Custom provider");

    customQueryClient.setQueryData(["provider-probe"], "custom-client");

    const view = render(
      <AgentServerUIProviders queryClient={customQueryClient} i18n={customI18n}>
        <CustomProbe />
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("query-client-kind")).toHaveTextContent("custom");
    expect(screen.getByTestId("query-client-value")).toHaveTextContent(
      "custom-client",
    );

    await waitFor(() => {
      expect(screen.getByTestId("translation-value")).toHaveTextContent(
        "Custom provider",
      );
      expect(
        screen.getByTestId("imperative-translation-value"),
      ).toHaveTextContent("Custom provider");
    });

    expect(getQueryClient()).toBe(customQueryClient);
    expect(getI18n()).toBe(customI18n);

    view.unmount();

    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());
  });

  it("passes disabled and runtime analytics configuration to TelemetryProvider", () => {
    telemetryProviderMock.mockClear();

    const noAnalyticsView = render(
      <AgentServerUIProviders>
        <div data-testid="child">child</div>
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("child")).toHaveTextContent("child");
    expect(telemetryProviderMock).toHaveBeenCalledWith(
      expect.objectContaining({ config: false }),
    );

    noAnalyticsView.unmount();
    telemetryProviderMock.mockClear();

    const analytics = {
      provider: "posthog" as const,
      apiKey: "phc_embedded",
      apiHost: "https://events.example.com",
      uiHost: "https://posthog.example.com",
    };

    render(
      <AgentServerUIProviders analytics={analytics}>
        <div data-testid="child-with-analytics">child</div>
      </AgentServerUIProviders>,
    );

    expect(telemetryProviderMock).toHaveBeenCalledWith(
      expect.objectContaining({
        config: {
          apiKey: analytics.apiKey,
          apiHost: analytics.apiHost,
          uiHost: analytics.uiHost,
        },
      }),
    );
  });

  it("wraps children in a scoped, customizable style root by default", () => {
    const { unmount } = render(
      <AgentServerUIProviders
        contentClassName="min-h-screen"
        styleOverrides={{ "--oh-color-base": "#010203" }}
      >
        <div data-testid="styled-child">child</div>
      </AgentServerUIProviders>,
    );

    const scopeRoot = document.querySelector<HTMLDivElement>(
      AGENT_SERVER_UI_SCOPE_SELECTOR,
    );

    expect(scopeRoot).toBeInTheDocument();
    expect(scopeRoot?.style.getPropertyValue("--oh-color-base")).toBe(
      "#010203",
    );

    const themedContainer =
      scopeRoot?.firstElementChild as HTMLDivElement | null;
    expect(themedContainer).toHaveAttribute("data-theme", "dark");
    expect(themedContainer).toHaveClass("dark", "min-h-screen");
    expect(themedContainer).toContainElement(
      screen.getByTestId("styled-child"),
    );

    unmount();

    render(
      <AgentServerUIProviders withStyleRoot={false}>
        <div data-testid="unstyled-child">child</div>
      </AgentServerUIProviders>,
    );

    expect(document.querySelector(AGENT_SERVER_UI_SCOPE_SELECTOR)).toBeNull();
  });

  it("exposes a standalone style root for host-controlled customization", () => {
    render(
      <AgentServerUIRoot
        className="outer-shell"
        contentClassName="inner-shell"
        theme="light"
        styleOverrides={{ "--oh-color-primary": "#abcdef" }}
      >
        <div data-testid="root-child">child</div>
      </AgentServerUIRoot>,
    );

    const scopeRoot = document.querySelector<HTMLDivElement>(
      AGENT_SERVER_UI_SCOPE_SELECTOR,
    );

    expect(scopeRoot).toHaveClass("outer-shell");
    expect(scopeRoot?.style.getPropertyValue("--oh-color-primary")).toBe(
      "#abcdef",
    );

    const themedContainer =
      scopeRoot?.firstElementChild as HTMLDivElement | null;
    expect(themedContainer).toHaveAttribute("data-theme", "light");
    expect(themedContainer).toHaveClass("light", "inner-shell");
    expect(themedContainer).toContainElement(screen.getByTestId("root-child"));
  });
});
