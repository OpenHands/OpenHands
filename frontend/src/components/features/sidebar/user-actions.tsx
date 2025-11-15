import React from "react";
import { UserAvatar } from "./user-avatar";
import { useMe } from "#/hooks/query/use-me";
import { useShouldShowUserFeatures } from "#/hooks/use-should-show-user-features";
import { useConfig } from "#/hooks/query/use-config";
import { AccountSettingsContextMenu } from "../context-menu/account-settings-context-menu";
import { UserContextMenu } from "../user/user-context-menu";
import { cn } from "#/utils/utils";

interface UserActionsProps {
  user?: { avatar_url: string };
  isLoading?: boolean;
}

export function UserActions({ user, isLoading }: UserActionsProps) {
  const { data: me } = useMe();
  const [accountContextMenuIsVisible, setAccountContextMenuIsVisible] =
    React.useState(false);

  const { data: config } = useConfig();

  // Use the shared hook to determine if user actions should be shown
  const shouldShowUserActions = useShouldShowUserFeatures();

  const toggleAccountMenu = () => {
    // Always toggle the menu, even if user is undefined
    setAccountContextMenuIsVisible((prev) => !prev);
  };

  const closeAccountMenu = () => {
    if (accountContextMenuIsVisible) {
      setAccountContextMenuIsVisible(false);
    }
  };

  const handleLogout = () => {
    closeAccountMenu();
  };

  const isOSS = config?.APP_MODE === "oss";

  // Show the menu based on the new logic
  const showMenu =
    accountContextMenuIsVisible && (shouldShowUserActions || isOSS);

  return (
    <div
      data-testid="user-actions"
      className="w-8 h-8 relative cursor-pointer group"
    >
      <UserAvatar
        avatarUrl={user?.avatar_url}
        onClick={toggleAccountMenu}
        isLoading={isLoading}
      />

      {(shouldShowUserActions || isOSS) && (
        <div
          className={cn(
            "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto",
            showMenu && "opacity-100 pointer-events-auto",
          )}
        >
          <AccountSettingsContextMenu
            onLogout={handleLogout}
            onClose={closeAccountMenu}
          />
        </div>
      )}
      {accountContextMenuIsVisible && !!user && (
        <div className="w-sm absolute left-[calc(100%+12px)] bottom-0 z-10">
          <UserContextMenu
            type={me?.role || "user"}
            onClose={closeAccountMenu}
          />
        </div>
      )}
    </div>
  );
}
