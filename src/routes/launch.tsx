import React from "react";
import { useNavigate, useSearchParams } from "react-router";
import { useTranslation } from "react-i18next";
import { PluginLaunchModal } from "#/components/features/launch/plugin-launch-modal";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { I18nKey } from "#/i18n/declaration";

type ErrorType = "no_plugins" | "invalid_format" | "creation_failed";

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
  onGoHome,
  onTryAgain,
}: {
  errorType: ErrorType;
  onGoHome: () => void;
  onTryAgain?: () => void;
}) {
  const { t } = useTranslation("openhands");

  const errorMessages: Record<ErrorType, string> = {
    no_plugins: t(I18nKey.LAUNCH$ERROR_NO_PLUGINS),
    invalid_format: t(I18nKey.LAUNCH$ERROR_INVALID_FORMAT),
    creation_failed: t(I18nKey.LAUNCH$ERROR_CREATION_FAILED),
  };

  return (
    <div
      className="flex h-full items-center justify-center"
      data-testid="launch-error"
    >
      <div className="max-w-md rounded-lg bg-[var(--oh-surface)] p-6 text-center">
        <h2 className="mb-4 text-xl font-medium text-red-400">
          {t(I18nKey.LAUNCH$ERROR_TITLE)}
        </h2>
        <p className="mb-6 text-[var(--oh-text-tertiary)]">
          {errorMessages[errorType]}
        </p>
        <div className="flex justify-center gap-3">
          <button
            type="button"
            onClick={onGoHome}
            className="rounded-md bg-tertiary px-4 py-2 text-sm text-[var(--oh-foreground)] hover:bg-[var(--oh-interactive-hover)]"
            data-testid="go-home-button"
          >
            {t(I18nKey.LAUNCH$GO_HOME)}
          </button>
          {onTryAgain && (
            <button
              type="button"
              onClick={onTryAgain}
              className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700"
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
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { t } = useTranslation("openhands");
  const createConversation = useCreateConversation();

  const [creationError, setCreationError] = React.useState<string | null>(null);

  const parseResult = React.useMemo(
    () => parsePluginsFromUrl(searchParams),
    [searchParams],
  );

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
      setCreationError(t(I18nKey.LAUNCH$ERROR_CREATION_UNKNOWN));
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

  // Show error if conversation creation failed
  if (creationError) {
    return (
      <ErrorDisplay
        errorType="creation_failed"
        onGoHome={handleGoHome}
        onTryAgain={handleTryAgain}
      />
    );
  }

  return (
    <div className="h-full" data-testid="launch-route">
      <PluginLaunchModal
        plugins={parseResult.plugins}
        message={parseResult.message}
        isLoading={createConversation.isPending}
        onStartConversation={handleStartConversation}
        onClose={handleGoHome}
      />
    </div>
  );
}
