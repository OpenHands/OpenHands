import React from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { PluginLaunchModal } from "#/components/features/launch/plugin-launch-modal";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { I18nKey } from "#/i18n/declaration";
import { useActiveBackend } from "#/contexts/active-backend-context";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  automationDetailPath,
  hasAutomationInterface,
} from "#/manifests/automation-interface";
import {
  configSchemaToFormFields,
  pluginSpecTemplateId,
  pluginSpecTemplateVersion,
} from "#/utils/plugin-launch-preflight";

type ErrorType =
  | "no_plugins"
  | "invalid_format"
  | "creation_failed"
  | "install_failed";

interface ParseResult {
  plugins: PluginSpec[];
  message?: string;
  error?: ErrorType;
}

function sanitizeMessage(message: string | null): string | undefined {
  if (!message) return undefined;
  // Strip ALL HTML tags for plain text display
  const stripped = message.replace(/<[^>]*>/g, "").slice(0, 500);
  return stripped || undefined;
}

function parsePluginsFromUrl(searchParams: URLSearchParams): ParseResult {
  // Try base64 encoded plugins parameter first (production format)
  const pluginsParam = searchParams.get("plugins");
  if (pluginsParam) {
    try {
      const decoded = atob(pluginsParam);
      const parsed = JSON.parse(decoded);

      if (!Array.isArray(parsed)) {
        return { plugins: [], error: "invalid_format" };
      }

      // Validate each plugin has at least a source
      const validPlugins: PluginSpec[] = [];
      for (const item of parsed) {
        if (
          typeof item !== "object" ||
          !item ||
          typeof item.source !== "string"
        ) {
          return { plugins: [], error: "invalid_format" };
        }
        validPlugins.push({
          source: item.source,
          ref: item.ref ?? null,
          repo_path: item.repo_path ?? null,
          parameters: item.parameters ?? null,
        });
      }

      if (validPlugins.length === 0) {
        return { plugins: [], error: "no_plugins" };
      }

      return {
        plugins: validPlugins,
        message: sanitizeMessage(searchParams.get("message")),
      };
    } catch {
      return { plugins: [], error: "invalid_format" };
    }
  }

  // Fallback: simple params format for dev/testing
  const pluginSource = searchParams.get("plugin_source");
  if (pluginSource) {
    const plugin: PluginSpec = {
      source: pluginSource,
      ref: searchParams.get("plugin_ref") ?? null,
      repo_path: searchParams.get("plugin_repo_path") ?? null,
      parameters: null,
    };
    return {
      plugins: [plugin],
      message: sanitizeMessage(searchParams.get("message")),
    };
  }

  return { plugins: [], error: "no_plugins" };
}

function ErrorDisplay({
  errorType,
  detail,
  onGoHome,
  onTryAgain,
}: {
  errorType: ErrorType;
  detail?: string | null;
  onGoHome: () => void;
  onTryAgain?: () => void;
}) {
  const { t } = useTranslation("openhands");

  const errorMessages: Record<ErrorType, string> = {
    no_plugins: t(I18nKey.LAUNCH$ERROR_NO_PLUGINS),
    invalid_format: t(I18nKey.LAUNCH$ERROR_INVALID_FORMAT),
    creation_failed: t(I18nKey.LAUNCH$ERROR_CREATION_FAILED),
    install_failed: t(I18nKey.LAUNCH$ERROR_INSTALL_UNKNOWN),
  };

  return (
    <div
      className="flex h-full items-center justify-center"
      data-testid="launch-error"
    >
      <div className="max-w-md rounded-lg bg-surface p-6 text-center">
        <h2 className="mb-4 text-xl font-medium text-feedback-error">
          {t(I18nKey.LAUNCH$ERROR_TITLE)}
        </h2>
        <p className="mb-6 text-text-tertiary">{errorMessages[errorType]}</p>
        {detail && (
          <p className="mb-6 break-words text-sm text-text-tertiary">
            {detail}
          </p>
        )}
        <div className="flex justify-center gap-3">
          <button
            type="button"
            onClick={onGoHome}
            className="rounded-md bg-tertiary px-4 py-2 text-sm text-foreground hover:bg-interactive-hover"
            data-testid="go-home-button"
          >
            {t(I18nKey.LAUNCH$GO_HOME)}
          </button>
          {onTryAgain && (
            <button
              type="button"
              onClick={onTryAgain}
              className="rounded-md bg-contrast px-4 py-2 text-sm text-contrast-foreground hover:bg-contrast/90"
              data-testid="try-again-button"
            >
              {t(I18nKey.LAUNCH$TRY_AGAIN)}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default function LaunchRoute() {
  const { backend } = useActiveBackend();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { t } = useTranslation("openhands");
  const createConversation = useCreateConversation();

  const [creationError, setCreationError] = React.useState<string | null>(null);
  const [creationErrorKind, setCreationErrorKind] = React.useState<
    "conversation" | "automation"
  >("conversation");
  const [isInstalling, setIsInstalling] = React.useState(false);

  const parseResult = React.useMemo(
    () => parsePluginsFromUrl(searchParams),
    [searchParams],
  );

  // `mode=automation` asks the route to install the plugin as an automation
  // (POST /plugin-spec) instead of starting a conversation. Requires an
  // admitted automation interface manifest, since submit navigates to the
  // automation detail page afterwards.
  const automationMode =
    searchParams.get("mode") === "automation" &&
    hasAutomationInterface() &&
    parseResult.plugins.length === 1;

  // Preflight (OpenHands/automation#630): fetch the manifest's config schema
  // and suggested trigger so the modal renders typed fields and an editable
  // trigger. Any failure means the backend predates the endpoint — the modal
  // silently falls back to the free-text-parameter flow, and the create
  // request still works (the service applies manifest defaults itself).
  const [preflight, setPreflight] = React.useState<
    | {
        description?: string | null;
        resolvedRef?: string | null;
        fields: ReturnType<typeof configSchemaToFormFields>;
        suggestedTrigger?: {
          type: string;
          schedule?: string;
          timezone?: string;
        } | null;
      }
    | null
    | undefined
  >(automationMode ? undefined : null);

  React.useEffect(() => {
    if (!automationMode) return;
    const plugin = parseResult.plugins[0];
    let cancelled = false;
    AutomationService.pluginSpecPreflight({
      source: plugin.source,
      ref: plugin.ref,
      repo_path: plugin.repo_path,
    })
      .then((response) => {
        if (cancelled) return;
        setPreflight({
          description: response.manifest.description ?? null,
          resolvedRef: response.resolved_ref ?? null,
          fields: configSchemaToFormFields(response),
          suggestedTrigger: response.suggested_trigger ?? null,
        });
      })
      .catch(() => {
        // Endpoint absent or unreachable: fall back to the plain flow.
        if (!cancelled) setPreflight(null);
      });
    return () => {
      cancelled = true;
    };
  }, [automationMode]);

  const handleStartConversation = async (
    plugins: PluginSpec[],
    initialMessage?: string,
  ) => {
    setCreationError(null);
    try {
      const result = await createConversation.mutateAsync({
        plugins,
        query: initialMessage,
        entryPoint: "launch_deeplink",
      });
      navigate(`/conversations/${result.conversation_id}`);
    } catch {
      setCreationErrorKind("conversation");
      setCreationError(t(I18nKey.LAUNCH$ERROR_CREATION_UNKNOWN));
    }
  };

  /**
   * Automation name derived from the plugin source, e.g.
   * `github:org/recipe` -> `recipe (org)`.
   */
  const automationNameForPlugin = (plugin: PluginSpec): string => {
    const source = plugin.source
      .replace(/^github:/, "")
      .replace(/^https?:\/\/github\.com\//, "")
      .replace(/\.git$/, "");
    const parts = source.split("/");
    const repo = parts[parts.length - 1] || plugin.source;
    const owner = parts.length > 1 ? parts[0] : null;
    return owner ? `${repo} (${owner})` : repo;
  };

  const handleInstallAutomation = async (
    plugins: PluginSpec[],
    trigger?: { type: "cron"; schedule: string; timezone: string } | null,
  ) => {
    const plugin = plugins[0];
    setCreationError(null);
    setIsInstalling(true);
    try {
      const created = await AutomationService.createPluginSpecAutomation({
        name: automationNameForPlugin(plugin),
        plugin: {
          source: plugin.source,
          ref: plugin.ref ?? undefined,
          repo_path: plugin.repo_path ?? undefined,
        },
        config: (plugin.parameters ?? undefined) as
          | Record<string, unknown>
          | undefined,
        trigger,
        // Dedupe repeated badge clicks: the service returns the existing
        // automation (HTTP 200) when this template id is already installed.
        template: {
          id: pluginSpecTemplateId(plugin),
          version: pluginSpecTemplateVersion(preflight?.resolvedRef, plugin),
        },
        enabled: true,
      });
      navigate(automationDetailPath(created.id));
    } catch (error) {
      setCreationErrorKind("automation");
      setCreationError(
        error instanceof Error
          ? error.message
          : t(I18nKey.LAUNCH$ERROR_INSTALL_UNKNOWN),
      );
    } finally {
      setIsInstalling(false);
    }
  };

  const handleGoHome = () => {
    navigate("/conversations");
  };

  const handleTryAgain = () => {
    setCreationError(null);
    window.location.reload();
  };

  // Show error if parsing failed
  if (parseResult.error) {
    return (
      <ErrorDisplay errorType={parseResult.error} onGoHome={handleGoHome} />
    );
  }

  // Show error if conversation creation or automation install failed
  if (creationError) {
    return (
      <ErrorDisplay
        errorType={
          creationErrorKind === "automation"
            ? "install_failed"
            : "creation_failed"
        }
        detail={creationError}
        onGoHome={handleGoHome}
        onTryAgain={handleTryAgain}
      />
    );
  }

  return (
    <div className="h-full" data-testid="launch-route">
      <PluginLaunchModal
        plugins={parseResult.plugins}
        parametersSupported={backend.kind !== "local"}
        message={parseResult.message}
        isLoading={createConversation.isPending || isInstalling}
        mode={automationMode ? "automation" : "conversation"}
        preflight={automationMode ? (preflight ?? null) : null}
        isPreflightLoading={automationMode && preflight === undefined}
        onStartConversation={handleStartConversation}
        onInstallAutomation={
          automationMode ? handleInstallAutomation : undefined
        }
        onClose={handleGoHome}
      />
    </div>
  );
}
