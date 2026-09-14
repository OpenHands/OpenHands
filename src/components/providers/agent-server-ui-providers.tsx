import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider } from "react-i18next";
import { type i18n as I18nInstance } from "i18next";
import {
  getDefaultQueryClient,
  getQueryClient,
  setQueryClient,
} from "#/query-client-config";
import {
  OPENHANDS_I18N_NAMESPACE,
  getDefaultI18n,
  getI18n,
  setI18n,
} from "#/i18n";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useHydrateFreeModels } from "#/hooks/query/use-free-models";
import type { TelemetryConfig } from "#/services/telemetry";
import { TelemetryProvider } from "./telemetry-provider";
import {
  AgentServerUIRoot,
  type AgentServerUIRootProps,
} from "./agent-server-ui-root";

export interface AgentServerUIPostHogAnalyticsConfig extends TelemetryConfig {
  provider: "posthog";
}

export type AgentServerUIAnalyticsConfig =
  | AgentServerUIPostHogAnalyticsConfig
  | false
  | null;

export const DEFAULT_AGENT_SERVER_ANALYTICS: AgentServerUIAnalyticsConfig = {
  provider: "posthog",
};

export interface AgentServerUIProvidersProps extends Pick<
  AgentServerUIRootProps,
  "className" | "contentClassName" | "style" | "styleOverrides" | "theme"
> {
  children: React.ReactNode;
  queryClient?: QueryClient;
  analytics?: AgentServerUIAnalyticsConfig;
  i18n?: I18nInstance;
  withStyleRoot?: boolean;
}

/**
 * Fetches the DB-driven free / default model flags once and mirrors them into
 * the free-models store. Rendered inside the query provider so leaf components
 * can read the flags synchronously without a QueryClientProvider in scope.
 */
function FreeModelsHydrator() {
  useHydrateFreeModels();
  return null;
}

export function AgentServerUIProviders({
  children,
  queryClient,
  analytics,
  i18n,
  className,
  contentClassName,
  style,
  styleOverrides,
  theme,
  withStyleRoot = true,
}: AgentServerUIProvidersProps) {
  const resolvedQueryClient = React.useMemo(
    () => queryClient ?? getDefaultQueryClient(),
    [queryClient],
  );
  const resolvedI18n = React.useMemo(() => i18n ?? getDefaultI18n(), [i18n]);
  const previousProvidersRef = React.useRef<{
    queryClient: QueryClient;
    i18n: I18nInstance;
  } | null>(null);

  if (!previousProvidersRef.current) {
    previousProvidersRef.current = {
      queryClient: getQueryClient(),
      i18n: getI18n(),
    };
  }

  setQueryClient(resolvedQueryClient);
  setI18n(resolvedI18n);

  React.useEffect(() => {
    // Strict Mode replays setup/cleanup without re-rendering in between, so
    // the render-phase assignment above does not run again after the cleanup
    // has handed the globals back. Re-install on every setup so this root's
    // instances stay active for imperative callers.
    setQueryClient(resolvedQueryClient);
    setI18n(resolvedI18n);

    return () => {
      const previous = previousProvidersRef.current;
      if (!previous) return;

      // Only hand the globals back if this root still owns them. A sibling
      // root that mounted afterwards is the current owner, and unmounting in
      // the other order must not clobber its instances.
      if (getQueryClient() === resolvedQueryClient) {
        setQueryClient(previous.queryClient);
      }
      if (getI18n() === resolvedI18n) {
        setI18n(previous.i18n);
      }
    };
  }, [resolvedQueryClient, resolvedI18n]);

  const posthogConfig =
    analytics && analytics.provider === "posthog"
      ? {
          apiKey: analytics.apiKey,
          apiHost: analytics.apiHost,
          uiHost: analytics.uiHost,
        }
      : false;
  const content = (
    <TelemetryProvider config={posthogConfig}>
      <FreeModelsHydrator />
      {children}
    </TelemetryProvider>
  );

  const wrappedContent = withStyleRoot ? (
    <AgentServerUIRoot
      className={className}
      contentClassName={contentClassName}
      style={style}
      styleOverrides={styleOverrides}
      theme={theme}
    >
      {content}
    </AgentServerUIRoot>
  ) : (
    content
  );

  return (
    <I18nextProvider i18n={resolvedI18n} defaultNS={OPENHANDS_I18N_NAMESPACE}>
      <QueryClientProvider client={resolvedQueryClient}>
        <ActiveBackendProvider>{wrappedContent}</ActiveBackendProvider>
      </QueryClientProvider>
    </I18nextProvider>
  );
}
