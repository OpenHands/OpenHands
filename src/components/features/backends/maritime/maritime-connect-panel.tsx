import React from "react";
import { useTranslation } from "react-i18next";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { KeyRound, Loader2, Moon, Plus, RefreshCw, Trash2 } from "lucide-react";

import {
  getMaritimeAgentPhase,
  getMaritimeErrorMessage,
  isOpenableAgent,
  requireMaritimeBridge,
  type MaritimeAgent,
  type MaritimeAgentPhase,
  type MaritimeAuthState,
} from "#/api/maritime/maritime-bridge";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ErrorAlert } from "#/components/shared/error-alert";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { MARITIME_QUERY_KEYS } from "#/hooks/query/query-keys";
import {
  useMaritimeAccount,
  useMaritimeAgents,
  useMaritimeAuthState,
} from "#/hooks/query/use-maritime";
import {
  useMaritimeAgentActions,
  type MaritimeBusyState,
} from "#/hooks/use-maritime-agent-actions";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { MaritimeNewAgentForm } from "./maritime-new-agent-form";
import { MaritimeSignIn } from "./maritime-sign-in";

const PANEL_CLASS =
  "flex min-h-[13.5rem] w-full flex-col rounded-xl border border-border";
const ICON_BUTTON_CLASS =
  "inline-flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-muted hover:bg-interactive-hover hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus disabled:cursor-not-allowed disabled:opacity-40";

const PHASE_LABEL_KEY: Record<MaritimeAgentPhase, I18nKey> = {
  running: I18nKey.MARITIME$STATUS_RUNNING,
  sleeping: I18nKey.MARITIME$STATUS_SLEEPING,
  starting: I18nKey.MARITIME$STATUS_STARTING,
  stopped: I18nKey.MARITIME$STATUS_STOPPED,
  failed: I18nKey.MARITIME$STATUS_FAILED,
};

const PHASE_DOT_CLASS: Record<MaritimeAgentPhase, string> = {
  running: "bg-status-success",
  sleeping: "bg-muted",
  starting: "bg-warning animate-pulse",
  stopped: "bg-muted",
  failed: "bg-status-error",
};

/** What a row shows while an action on it runs. */
function useBusyLabel(busy: MaritimeBusyState | null, agentId: string) {
  const { t } = useTranslation("openhands");
  if (!busy || !("agentId" in busy) || busy.agentId !== agentId) return null;
  switch (busy.kind) {
    case "opening":
    case "resetting":
      return busy.stage === "machine"
        ? t(I18nKey.MARITIME$STAGE_MACHINE)
        : t(I18nKey.MARITIME$STAGE_AGENT);
    case "sleeping":
      return t(I18nKey.MARITIME$STATUS_SLEEPING);
    case "deleting":
      return t(I18nKey.MARITIME$DELETING);
    default:
      return null;
  }
}

interface AgentRowProps {
  agent: MaritimeAgent;
  busy: MaritimeBusyState | null;
  isBusy: boolean;
  onOpen: () => void;
  onSleep: () => void;
  onDelete: () => void;
}

function AgentRow({
  agent,
  busy,
  isBusy,
  onOpen,
  onSleep,
  onDelete,
}: AgentRowProps) {
  const { t } = useTranslation("openhands");
  const phase = getMaritimeAgentPhase(agent.status);
  const busyLabel = useBusyLabel(busy, agent.id);
  const canOpen = isOpenableAgent(agent) && !isBusy;
  const memory = agent.memMb
    ? t(I18nKey.MARITIME$MEMORY_OPTION, {
        gb: Number((agent.memMb / 1024).toFixed(2)),
      })
    : null;

  return (
    <li
      data-testid={`maritime-agent-${agent.id}`}
      className="flex items-center gap-2 px-3 py-2"
    >
      <span
        aria-hidden
        className={cn("size-2 shrink-0 rounded-full", PHASE_DOT_CLASS[phase])}
      />
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm text-white">{agent.name}</span>
        <span className="truncate text-xs tabular-nums text-muted">
          {busyLabel ??
            [t(PHASE_LABEL_KEY[phase]), memory].filter(Boolean).join(" · ")}
        </span>
      </span>
      {busyLabel ? (
        <Loader2
          className="size-4 shrink-0 animate-spin text-muted"
          aria-hidden
        />
      ) : null}
      {phase === "running" ? (
        <button
          type="button"
          onClick={onSleep}
          disabled={isBusy}
          aria-label={t(I18nKey.MARITIME$SLEEP_AGENT, { name: agent.name })}
          title={t(I18nKey.MARITIME$SLEEP)}
          data-testid={`maritime-agent-sleep-${agent.id}`}
          className={ICON_BUTTON_CLASS}
        >
          <Moon className="size-4" aria-hidden />
        </button>
      ) : null}
      <button
        type="button"
        onClick={onDelete}
        disabled={isBusy}
        aria-label={t(I18nKey.MARITIME$DELETE_AGENT, { name: agent.name })}
        title={t(I18nKey.BUTTON$DELETE)}
        data-testid={`maritime-agent-delete-${agent.id}`}
        className={ICON_BUTTON_CLASS}
      >
        <Trash2 className="size-4" aria-hidden />
      </button>
      <BrandButton
        type="button"
        variant="secondary"
        onClick={onOpen}
        isDisabled={!canOpen}
        testId={`maritime-agent-open-${agent.id}`}
        ariaLabel={t(I18nKey.MARITIME$OPEN_AGENT, { name: agent.name })}
        startContent={
          agent.hasAccessKey ? null : (
            <KeyRound className="size-3.5" aria-hidden />
          )
        }
      >
        {t(I18nKey.MARITIME$OPEN)}
      </BrandButton>
    </li>
  );
}

/** One line saying where the API key is kept, so a restart holds no surprise. */
function StorageNote({ authState }: { authState: MaritimeAuthState }) {
  const { t } = useTranslation("openhands");
  const { storage } = authState;
  if (!authState.isSaved) {
    return (
      <p
        data-testid="maritime-key-storage"
        className="w-full text-pretty text-warning"
      >
        {t(I18nKey.MARITIME$NOT_SAVED)}
      </p>
    );
  }
  return (
    <p
      data-testid="maritime-key-storage"
      className="w-full truncate text-muted"
    >
      {storage.kind === "encrypted-file"
        ? t(I18nKey.MARITIME$SAVED_ENCRYPTED, { host: storage.host })
        : t(I18nKey.MARITIME$SAVED_IN_STORE, {
            store: storage.label,
            host: storage.host,
          })}
    </p>
  );
}

type PendingConfirmation =
  | { kind: "delete"; agent: MaritimeAgent }
  | { kind: "reset"; agent: MaritimeAgent };

interface MaritimeConnectPanelProps {
  /** Called once an agent is open and active, e.g. to close the modal. */
  onConnected?: () => void;
}

/**
 * The Maritime provider: sign in with an API key, then open one of the
 * account's OpenHands agents or create a new one. Opening an agent makes its
 * agent-server the active backend.
 */
export function MaritimeConnectPanel({
  onConnected,
}: MaritimeConnectPanelProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { data: authState, isLoading: isAuthLoading } = useMaritimeAuthState();
  const isSignedIn = authState?.isSignedIn === true;
  const account = useMaritimeAccount(isSignedIn);
  const agents = useMaritimeAgents(isSignedIn);
  const actions = useMaritimeAgentActions({ onOpened: () => onConnected?.() });
  const [isCreating, setIsCreating] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [pending, setPending] = React.useState<PendingConfirmation | null>(
    null,
  );

  const describe = React.useCallback(
    (cause: unknown) =>
      getMaritimeErrorMessage(cause) ?? t(I18nKey.MARITIME$ERROR_GENERIC),
    [t],
  );

  const signIn = useMutation({
    mutationFn: (apiKey: string) =>
      requireMaritimeBridge().saveApiKey({ apiKey }),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: MARITIME_QUERY_KEYS.all }),
    meta: { disableToast: true },
  });

  const signOut = useMutation({
    mutationFn: () => requireMaritimeBridge().signOut(),
    onSuccess: () => {
      setIsCreating(false);
      return queryClient.invalidateQueries({
        queryKey: MARITIME_QUERY_KEYS.all,
      });
    },
    meta: { disableToast: true },
  });

  /** Runs an action and turns a failure into the panel's error line. */
  const attempt = React.useCallback(
    async (action: () => Promise<unknown>) => {
      setError(null);
      try {
        await action();
        return true;
      } catch (cause) {
        setError(describe(cause));
        return false;
      }
    },
    [describe],
  );

  if (isAuthLoading) {
    return (
      <div className={cn(PANEL_CLASS, "items-center justify-center")}>
        <Loader2 className="size-5 animate-spin text-muted" aria-hidden />
      </div>
    );
  }

  if (!isSignedIn || !authState) {
    return (
      <div className={PANEL_CLASS}>
        <MaritimeSignIn
          isSaving={signIn.isPending}
          error={signIn.error ? describe(signIn.error) : null}
          onSaveApiKey={(apiKey) => signIn.mutate(apiKey)}
        />
      </div>
    );
  }

  const openAgent = (agent: MaritimeAgent) => {
    if (!agent.hasAccessKey) {
      setPending({ kind: "reset", agent });
      return;
    }
    void attempt(() => actions.open(agent));
  };

  const confirmPending = async () => {
    if (!pending) return;
    const { kind, agent } = pending;
    setPending(null);
    await attempt(() =>
      kind === "delete" ? actions.remove(agent) : actions.resetAccessKey(agent),
    );
  };

  const creating = actions.busy?.kind === "creating" ? actions.busy : null;
  const createProgress = creating
    ? t(
        creating.stage === "machine"
          ? I18nKey.MARITIME$CREATING
          : I18nKey.MARITIME$STAGE_AGENT,
      )
    : null;
  const agentList = agents.data ?? [];
  const plan = account.data?.plan;

  return (
    <div className={PANEL_CLASS} data-testid="maritime-connect-panel">
      <div
        data-testid="maritime-account"
        className="m-3 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border border-border px-3 py-2 text-xs"
      >
        <span className="flex min-w-0 flex-1 items-center gap-2 truncate text-white">
          <span
            aria-hidden
            className="size-1.5 rounded-full bg-status-success"
          />
          {plan
            ? t(I18nKey.MARITIME$SIGNED_IN_PLAN, { plan })
            : t(I18nKey.MARITIME$SIGNED_IN)}
          {account.data?.maxAwakeMachines ? (
            <span className="text-muted">
              {t(I18nKey.MARITIME$AWAKE_USAGE, {
                awake: account.data.awakeMachines,
                max: account.data.maxAwakeMachines,
              })}
            </span>
          ) : null}
        </span>
        <button
          type="button"
          onClick={() => signOut.mutate()}
          disabled={actions.isBusy || signOut.isPending}
          data-testid="maritime-sign-out"
          className="cursor-pointer text-muted hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
        >
          {t(I18nKey.MARITIME$SIGN_OUT)}
        </button>
        <StorageNote authState={authState} />
      </div>

      <div className="flex items-center justify-between gap-2 px-4 pb-2">
        <h3 className="text-sm font-medium text-white">
          {t(I18nKey.MARITIME$AGENTS_TITLE)}
        </h3>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => void agents.refetch()}
            disabled={agents.isFetching}
            aria-label={t(I18nKey.MARITIME$REFRESH)}
            title={t(I18nKey.MARITIME$REFRESH)}
            data-testid="maritime-refresh"
            className={ICON_BUTTON_CLASS}
          >
            <RefreshCw
              className={cn("size-4", agents.isFetching && "animate-spin")}
              aria-hidden
            />
          </button>
          {isCreating ? null : (
            <BrandButton
              type="button"
              variant="primary"
              onClick={() => {
                setError(null);
                setIsCreating(true);
              }}
              isDisabled={actions.isBusy || !account.data}
              testId="maritime-new-agent"
              startContent={<Plus className="size-4" aria-hidden />}
            >
              {t(I18nKey.MARITIME$NEW_AGENT)}
            </BrandButton>
          )}
        </div>
      </div>

      {isCreating && account.data ? (
        <MaritimeNewAgentForm
          account={account.data}
          isCreating={creating !== null}
          progress={createProgress}
          error={error}
          onCancel={() => {
            setError(null);
            setIsCreating(false);
          }}
          onSubmit={(input) => {
            void attempt(() => actions.create(input)).then((created) => {
              if (created) setIsCreating(false);
            });
          }}
        />
      ) : null}

      {agents.isLoading ? (
        <div className="flex flex-1 items-center justify-center py-6">
          <Loader2 className="size-5 animate-spin text-muted" aria-hidden />
        </div>
      ) : null}

      {agents.isError ? (
        <ErrorAlert testId="maritime-agents-error" className="mx-4 mb-3">
          {describe(agents.error)}
        </ErrorAlert>
      ) : null}

      {agents.isSuccess && agentList.length === 0 && !isCreating ? (
        <p
          data-testid="maritime-no-agents"
          className="px-4 py-6 text-center text-sm text-muted"
        >
          {t(I18nKey.MARITIME$NO_AGENTS)}
        </p>
      ) : null}

      {agentList.length > 0 ? (
        <ul
          data-testid="maritime-agents"
          className="mb-2 divide-y divide-border"
        >
          {agentList.map((agent) => (
            <AgentRow
              key={agent.id}
              agent={agent}
              busy={actions.busy}
              isBusy={actions.isBusy}
              onOpen={() => openAgent(agent)}
              onSleep={() => void attempt(() => actions.sleep(agent))}
              onDelete={() => setPending({ kind: "delete", agent })}
            />
          ))}
        </ul>
      ) : null}

      {error && !isCreating ? (
        <ErrorAlert testId="maritime-error" className="mx-4 mb-3">
          {error}
        </ErrorAlert>
      ) : null}

      {pending ? (
        <ConfirmationModal
          text={
            pending.kind === "delete"
              ? t(I18nKey.MARITIME$DELETE_CONFIRM, { name: pending.agent.name })
              : t(I18nKey.MARITIME$RESET_KEY_CONFIRM, {
                  name: pending.agent.name,
                })
          }
          confirmText={
            pending.kind === "delete"
              ? t(I18nKey.BUTTON$DELETE)
              : t(I18nKey.MARITIME$RESET_KEY)
          }
          onConfirm={() => void confirmPending()}
          onCancel={() => setPending(null)}
        />
      ) : null}
    </div>
  );
}
