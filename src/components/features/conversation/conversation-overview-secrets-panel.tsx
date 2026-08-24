import React, { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { useDeleteSecret } from "#/hooks/mutation/use-delete-secret";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { cn } from "#/utils/utils";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { useConversationOverviewDrawerOptional } from "./conversation-overview-drawer-context";

interface ConversationOverviewSecretsPanelProps {
  openAdd: boolean;
}

export function ConversationOverviewSecretsPanel({
  openAdd,
}: ConversationOverviewSecretsPanelProps) {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");
  const addRequestKey =
    useConversationOverviewDrawerOptional()?.addRequestKey ?? 0;

  const { data: secrets, isLoading } = useSearchSecrets();
  const { mutate: deleteSecret, isPending: isDeleting } = useDeleteSecret();

  const [view, setView] = useState<"list" | "add-secret-form">("list");
  const [secretToDelete, setSecretToDelete] = useState<string | null>(null);

  useEffect(() => {
    if (!openAdd) {
      return;
    }
    setView("add-secret-form");
  }, [openAdd]);

  useEffect(() => {
    if (addRequestKey === 0) {
      return;
    }
    setView("add-secret-form");
  }, [addRequestKey]);

  const invalidateSecrets = () => {
    queryClient.invalidateQueries({ queryKey: ["secrets-search"] });
    queryClient.invalidateQueries({ queryKey: ["secrets"] });
  };

  if (view === "add-secret-form") {
    return (
      <div
        data-testid="conversation-overview-secrets-add-form"
        className="flex h-full min-h-0 flex-col overflow-y-auto p-4"
      >
        <SecretForm
          mode="add"
          selectedSecret={null}
          onCancel={() => setView("list")}
        />
      </div>
    );
  }

  return (
    <div
      data-testid="conversation-overview-secrets-panel"
      className="flex h-full min-h-0 flex-col"
    >
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        {isLoading ? (
          <p className="px-2 py-4 text-sm text-[var(--oh-muted)]">
            {t(I18nKey.HOME$LOADING)}
          </p>
        ) : secrets && secrets.length > 0 ? (
          <ul className="flex flex-col gap-0.5">
            {secrets.map((secret) => (
              <li
                key={secret.name}
                data-testid="conversation-overview-secret-item"
                className="group flex items-center gap-2 rounded-md px-2 py-2 hover:bg-white/5"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-[var(--oh-foreground)]">
                    {secret.name}
                  </p>
                  {secret.description ? (
                    <p className="truncate text-xs text-[var(--oh-muted)]">
                      {secret.description}
                    </p>
                  ) : null}
                </div>
                <button
                  type="button"
                  data-testid={`conversation-overview-secret-delete-${secret.name}`}
                  aria-label={t(I18nKey.BUTTON$DELETE)}
                  onClick={() => setSecretToDelete(secret.name)}
                  className={cn(
                    "shrink-0 rounded-md p-1 text-[var(--oh-muted)]",
                    "opacity-0 transition-opacity group-hover:opacity-100",
                    "hover:bg-white/10 hover:text-[var(--oh-foreground)]",
                  )}
                >
                  <Trash2 className="size-3.5" aria-hidden />
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p
            data-testid="conversation-overview-secrets-empty"
            className={cn(
              extensionModuleEmptyStateClassName,
              "px-2 py-6 text-center text-sm",
            )}
          >
            {t(I18nKey.SECRETS$EMPTY)}
          </p>
        )}
      </div>

      {secretToDelete ? (
        <ConfirmationModal
          text={t(I18nKey.SECRETS$CONFIRM_DELETE_KEY)}
          isConfirming={isDeleting}
          onCancel={() => setSecretToDelete(null)}
          onConfirm={() => {
            deleteSecret(secretToDelete, {
              onSettled: () => setSecretToDelete(null),
              onSuccess: invalidateSecrets,
              onError: invalidateSecrets,
            });
          }}
        />
      ) : null}
    </div>
  );
}
