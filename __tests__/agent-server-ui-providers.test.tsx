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
  setI18n,
  setQueryClient,
} from "#/index";
import { createAgentServerQueryClient } from "#/query-client-config";
import * as ToastHandlers from "#/utils/custom-toast-handlers";
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

const createSiblingRoot = async (probeId: string, label: string) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const instance = await createTestI18n(label);
  client.setQueryData(["provider-probe"], `probe-${probeId}`);

  return render(
    <AgentServerUIProviders queryClient={client} i18n={instance}>
      <SiblingProbe probeId={probeId} />
    </AgentServerUIProviders>,
  );
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

  it.each([
    { unmountFirst: "1" as const, remaining: "2" as const },
    { unmountFirst: "2" as const, remaining: "1" as const },
  ])(
    "keeps sibling providers independent when root $unmountFirst unmounts first",
    async ({ unmountFirst, remaining }) => {
      const roots = {
        "1": await createSiblingRoot("1", "Root 1"),
        "2": await createSiblingRoot("2", "Root 2"),
      };

      expect(screen.getByTestId("client-1")).toHaveTextContent("probe-1");
      expect(screen.getByTestId("client-2")).toHaveTextContent("probe-2");

      await waitFor(() => {
        expect(screen.getByTestId("i18n-1")).toHaveTextContent("Root 1");
        expect(screen.getByTestId("i18n-2")).toHaveTextContent("Root 2");
      });

      roots[unmountFirst].unmount();
      expect(screen.getByTestId(`client-${remaining}`)).toHaveTextContent(
        `probe-${remaining}`,
      );
      expect(screen.getByTestId(`i18n-${remaining}`)).toHaveTextContent(
        `Root ${remaining}`,
      );

      roots[remaining].unmount();
      expect(getQueryClient()).toBe(getDefaultQueryClient());
      expect(getI18n()).toBe(getDefaultI18n());
    },
  );

  it("async consumers capture their originating runtime rather than a mutable singleton", async () => {
    const client1 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const client2 = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const i18n1 = await createTestI18n("Async root 1");
    const i18n2 = await createTestI18n("Async root 2");
    client1.setQueryData(["async-probe"], "client-1-value");
    client2.setQueryData(["async-probe"], "client-2-value");

    let resolvedResult: { queryData: unknown; label: string } | null = null;

    const AsyncConsumer = () => {
      const client = useQueryClient();
      const { t } = useTranslation(OPENHANDS_I18N_NAMESPACE);
      const triggerAsync = React.useCallback(async () => {
        await new Promise((resolve) => setTimeout(resolve, 50));
        resolvedResult = {
          queryData: client.getQueryData(["async-probe"]),
          label: t("PROVIDER$LABEL"),
        };
      }, [client, t]);

      React.useEffect(() => {
        void triggerAsync();
      }, [triggerAsync]);

      return <div data-testid="async-consumer" />;
    };

    const root1 = render(
      <AgentServerUIProviders queryClient={client1} i18n={i18n1}>
        <AsyncConsumer />
      </AgentServerUIProviders>,
    );

    const root2 = render(
      <AgentServerUIProviders queryClient={client2} i18n={i18n2}>
        <div data-testid="sibling" />
      </AgentServerUIProviders>,
    );
    root2.unmount();

    await waitFor(() => {
      expect(resolvedResult).toEqual({
        queryData: "client-1-value",
        label: "Async root 1",
      });
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

  it("query-cache toasts use the i18n bound to each dedicated query client", async () => {
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client1 = createAgentServerQueryClient();
    const client2 = createAgentServerQueryClient();
    const i18n1 = await createTestI18n("Async root 1");
    const i18n2 = await createTestI18n("Async root 2");
    i18n1.addResource(
      "en",
      OPENHANDS_I18N_NAMESPACE,
      "ERROR$GENERIC",
      "Root 1 generic",
    );
    i18n2.addResource(
      "en",
      OPENHANDS_I18N_NAMESPACE,
      "ERROR$GENERIC",
      "Root 2 generic",
    );

    const failQuery = (client: QueryClient, id: string) =>
      client.fetchQuery({
        queryKey: ["toast-probe", id],
        retry: false,
        queryFn: async () => {
          throw new Error("");
        },
      });

    vi.useFakeTimers();
    try {
      const root1 = render(
        <AgentServerUIProviders queryClient={client1} i18n={i18n1}>
          <div data-testid="root-1" />
        </AgentServerUIProviders>,
      );
      const root2 = render(
        <AgentServerUIProviders queryClient={client2} i18n={i18n2}>
          <div data-testid="root-2" />
        </AgentServerUIProviders>,
      );

      await expect(failQuery(client1, "1")).rejects.toBeInstanceOf(Error);

      expect(toast).toHaveBeenCalledWith("Root 1 generic", i18n1);
      expect(getI18n()).toBe(getDefaultI18n());

      root1.unmount();
      toast.mockClear();
      await vi.advanceTimersByTimeAsync(3000);

      await expect(failQuery(client2, "2")).rejects.toBeInstanceOf(Error);

      expect(toast).toHaveBeenCalledWith("Root 2 generic", i18n2);
      expect(getI18n()).toBe(getDefaultI18n());
      root2.unmount();
    } finally {
      vi.useRealTimers();
    }
  });
});
