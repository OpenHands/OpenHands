import React from "react";
import { useTranslation } from "react-i18next";
import { Trash2 } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { Typography } from "#/ui/typography";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import {
  settingsListIconActionButtonClassName,
  settingsListScrollContainerClassName,
  settingsListTableCellClassName,
  settingsListTableHeadClassName,
  settingsListTableHeaderCellClassName,
  settingsListTableRowClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import { cn } from "#/utils/utils";
import { useAppLoginStatus } from "#/hooks/query/use-app-login";
import {
  useAppLoginUsers,
  useCreateAppLoginUser,
  useDeleteAppLoginUser,
} from "#/hooks/query/use-app-login-users";
import { AppLoginService } from "#/api/app-login-service";
import { useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { APP_LOGIN_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useInvalidatePentestCapabilities } from "#/hooks/use-pentest-capabilities";

export const handle = { hideTitle: false };

export function UsersSettingsScreen() {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const invalidatePentestCapabilities = useInvalidatePentestCapabilities();
  const statusQuery = useAppLoginStatus();
  const enabled = statusQuery.data?.enabled === true;
  const usersQuery = useAppLoginUsers(enabled);
  const createUser = useCreateAppLoginUser();
  const deleteUser = useDeleteAppLoginUser();

  const [username, setUsername] = React.useState("");
  const [password, setPassword] = React.useState("");
  const [pendingDelete, setPendingDelete] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (statusQuery.isSuccess && !enabled) {
      navigate("/settings/app");
    }
  }, [enabled, navigate, statusQuery.isSuccess]);

  const canCreate =
    username.trim().length > 0 && password.length >= 4 && !createUser.isPending;

  const handleCreate = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canCreate) return;
    try {
      await createUser.mutateAsync({
        username: username.trim(),
        password,
      });
      displaySuccessToast(t(I18nKey.SETTINGS$USERS_CREATED));
      setUsername("");
      setPassword("");
    } catch (err) {
      displayErrorToast(
        err instanceof Error ? err.message : t(I18nKey.APP_LOGIN$ERROR),
      );
    }
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    try {
      await deleteUser.mutateAsync(pendingDelete);
      displaySuccessToast(t(I18nKey.SETTINGS$USERS_DELETED));
    } catch (err) {
      displayErrorToast(
        err instanceof Error ? err.message : t(I18nKey.APP_LOGIN$ERROR),
      );
    } finally {
      setPendingDelete(null);
    }
  };

  const handleLogout = async () => {
    await AppLoginService.logout();
    invalidatePentestCapabilities();
    await queryClient.invalidateQueries({
      queryKey: APP_LOGIN_QUERY_KEYS.session,
    });
  };

  if (statusQuery.isPending || (enabled && usersQuery.isPending)) {
    return (
      <div className="flex justify-center py-10">
        <LoadingSpinner size="small" />
      </div>
    );
  }

  if (!enabled) {
    return null;
  }

  const users = usersQuery.data ?? [];

  return (
    <div data-testid="users-settings-screen" className="flex flex-col gap-6">
      <form
        className="flex flex-col gap-4 rounded-xl border border-[var(--oh-border)] p-4"
        onSubmit={handleCreate}
      >
        <Typography.H3>{t(I18nKey.SETTINGS$USERS_ADD)}</Typography.H3>
        <div className="grid gap-4 md:grid-cols-2">
          <SettingsInput
            testId="users-settings-username"
            name="new-username"
            label={t(I18nKey.SETTINGS$USERS_USERNAME)}
            type="text"
            value={username}
            onChange={setUsername}
            required
          />
          <SettingsInput
            testId="users-settings-password"
            name="new-password"
            label={t(I18nKey.SETTINGS$USERS_PASSWORD)}
            type="password"
            value={password}
            onChange={setPassword}
            required
          />
        </div>
        <div className="flex justify-end">
          <BrandButton
            testId="users-settings-create"
            type="submit"
            variant="primary"
            isDisabled={!canCreate}
          >
            {t(I18nKey.SETTINGS$USERS_CREATE)}
          </BrandButton>
        </div>
      </form>

      <div className={settingsListScrollContainerClassName}>
        <div className={settingsListTableHeadClassName}>
          <div className={settingsListTableHeaderCellClassName}>
            {t(I18nKey.SETTINGS$USERS_USERNAME)}
          </div>
          <div className={settingsListTableHeaderCellClassName} />
        </div>

        {users.length === 0 ? (
          <div
            data-testid="users-settings-empty"
            className={extensionModuleEmptyStateClassName}
          >
            {t(I18nKey.SETTINGS$USERS_EMPTY)}
          </div>
        ) : (
          users.map((user) => (
            <div
              key={user.username}
              data-testid={`users-settings-row-${user.username}`}
              className={settingsListTableRowClassName}
            >
              <div className={settingsListTableCellClassName}>
                {user.username}
              </div>
              <div
                className={cn(settingsListTableCellClassName, "justify-end")}
              >
                <button
                  type="button"
                  data-testid={`users-settings-delete-${user.username}`}
                  className={settingsListIconActionButtonClassName}
                  aria-label={t(I18nKey.SETTINGS$USERS_DELETE)}
                  onClick={() => setPendingDelete(user.username)}
                  disabled={users.length <= 1 || deleteUser.isPending}
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="flex justify-end">
        <BrandButton
          testId="users-settings-logout"
          type="button"
          variant="secondary"
          onClick={handleLogout}
        >
          {t(I18nKey.SETTINGS$USERS_LOGOUT)}
        </BrandButton>
      </div>

      {pendingDelete && (
        <ConfirmationModal
          text={t(I18nKey.SETTINGS$USERS_DELETE_CONFIRM, {
            username: pendingDelete,
          })}
          onConfirm={handleConfirmDelete}
          onCancel={() => setPendingDelete(null)}
          isConfirming={deleteUser.isPending}
        />
      )}
    </div>
  );
}

export default UsersSettingsScreen;
