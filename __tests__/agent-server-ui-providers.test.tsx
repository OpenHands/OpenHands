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
        {String(currentQueryClient.getQueryData(["provider-probe"]))}
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

  it("injects a custom query client and i18n instance scoped to the provider tree without polluting globals", async () => {
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
    });

    // Standalone globals remain unpolluted
    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());

    view.unmount();

    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());
  });

  it("keeps custom instances stable through Strict Mode replay", async () => {
    const customQueryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    const customI18n = await createTestI18n("Strict mode provider");
    customQueryClient.setQueryData(["provider-probe"], "strict-client");

    const view = render(
      <React.StrictMode>
        <AgentServerUIProviders
          queryClient={customQueryClient}
          i18n={customI18n}
        >
          <CustomProbe />
        </AgentServerUIProviders>
      </React.StrictMode>,
    );

    expect(screen.getByTestId("query-client-kind")).toHaveTextContent("custom");
    expect(screen.getByTestId("query-client-value")).toHaveTextContent(
      "strict-client",
    );

    await waitFor(() => {
      expect(screen.getByTestId("translation-value")).toHaveTextContent(
        "Strict mode provider",
      );
    });

    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());

    view.unmount();

    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());
  });

  it("keeps two sibling providers independent across every unmount order", async () => {
    const client1 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const client2 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const i18n1 = await createTestI18n("Root 1");
    const i18n2 = await createTestI18n("Root 2");

    client1.setQueryData(["provider-probe"], "probe-1");
    client2.setQueryData(["provider-probe"], "probe-2");

    const SiblingProbe = ({ probeId }: { probeId: string }) => {
      const client = useQueryClient();
      const { t } = useTranslation(OPENHANDS_I18N_NAMESPACE);
      return (
        <div>
          <div data-testid={`client-${probeId}`}>
            {String(client.getQueryData(["provider-probe"]))}
          </div>
          <div data-testid={`i18n-${probeId}`}>{t("PROVIDER$LABEL")}</div>
        </div>
      );
    };

    const root1 = render(
      <AgentServerUIProviders queryClient={client1} i18n={i18n1}>
        <SiblingProbe probeId="1" />
      </AgentServerUIProviders>,
    );
    const root2 = render(
      <AgentServerUIProviders queryClient={client2} i18n={i18n2}>
        <SiblingProbe probeId="2" />
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("client-1")).toHaveTextContent("probe-1");
    expect(screen.getByTestId("client-2")).toHaveTextContent("probe-2");

    await waitFor(() => {
      expect(screen.getByTestId("i18n-1")).toHaveTextContent("Root 1");
      expect(screen.getByTestId("i18n-2")).toHaveTextContent("Root 2");
    });

    // Unmount Root 1 first: Root 2 is unaffected
    root1.unmount();
    expect(screen.getByTestId("client-2")).toHaveTextContent("probe-2");
    expect(screen.getByTestId("i18n-2")).toHaveTextContent("Root 2");

    // Unmount Root 2: standalone globals remain unaffected
    root2.unmount();
    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());
  });

  it("leaves Root 1 intact when Root 2 unmounts first", async () => {
    const client1 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const client2 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const i18n1 = await createTestI18n("Root 1");
    const i18n2 = await createTestI18n("Root 2");

    client1.setQueryData(["provider-probe"], "probe-1");
    client2.setQueryData(["provider-probe"], "probe-2");

    const SiblingProbe = ({ probeId }: { probeId: string }) => {
      const client = useQueryClient();
      const { t } = useTranslation(OPENHANDS_I18N_NAMESPACE);
      return (
        <div>
          <div data-testid={`client-${probeId}`}>
            {String(client.getQueryData(["provider-probe"]))}
          </div>
          <div data-testid={`i18n-${probeId}`}>{t("PROVIDER$LABEL")}</div>
        </div>
      );
    };

    const root1 = render(
      <AgentServerUIProviders queryClient={client1} i18n={i18n1}>
        <SiblingProbe probeId="1" />
      </AgentServerUIProviders>,
    );
    const root2 = render(
      <AgentServerUIProviders queryClient={client2} i18n={i18n2}>
        <SiblingProbe probeId="2" />
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("client-1")).toHaveTextContent("probe-1");
    expect(screen.getByTestId("client-2")).toHaveTextContent("probe-2");

    // Unmount Root 2 first: Root 1 is unaffected
    root2.unmount();
    expect(screen.getByTestId("client-1")).toHaveTextContent("probe-1");
    expect(screen.getByTestId("i18n-1")).toHaveTextContent("Root 1");

    root1.unmount();
    expect(getQueryClient()).toBe(getDefaultQueryClient());
    expect(getI18n()).toBe(getDefaultI18n());
  });

  it("async consumers capture their originating runtime rather than a mutable singleton", async () => {
    const client1 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const client2 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    client1.setQueryData(["async-probe"], "client-1-value");
    client2.setQueryData(["async-probe"], "client-2-value");

    let resolvedResult: unknown = null;

    const AsyncConsumer = () => {
      const client = useQueryClient();
      const triggerAsync = React.useCallback(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
        resolvedResult = client.getQueryData(["async-probe"]);
      }, [client]);

      React.useEffect(() => {
        void triggerAsync();
      }, [triggerAsync]);

      return <div data-testid="async-consumer" />;
    };

    const root1 = render(
      <AgentServerUIProviders queryClient={client1}>
        <AsyncConsumer />
      </AgentServerUIProviders>,
    );

    // Mount and unmount Root 2 while Root 1's async operation is in-flight
    const root2 = render(
      <AgentServerUIProviders queryClient={client2}>
        <div data-testid="sibling" />
      </AgentServerUIProviders>,
    );
    root2.unmount();

    await waitFor(() => {
      expect(resolvedResult).toBe("client-1-value");
    });

    root1.unmount();
  });

  it("retains deterministic standalone compatibility without provider-driven global swapping", async () => {
    const customDefaultClient = new QueryClient();
    setQueryClient(customDefaultClient);

    const embeddedClient = new QueryClient();
    const embeddedI18n = await createTestI18n("Embedded");

    const view = render(
      <AgentServerUIProviders queryClient={embeddedClient} i18n={embeddedI18n}>
        <CustomProbe />
      </AgentServerUIProviders>,
    );

    expect(screen.getByTestId("query-client-kind")).toHaveTextContent("custom");
    // Imperative / standalone callers see their own explicit global instance, not the embedded one
    expect(getQueryClient()).toBe(customDefaultClient);

    view.unmount();
    expect(getQueryClient()).toBe(customDefaultClient);
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
