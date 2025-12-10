import React from "react";
import { UserAvatar } from "./user-avatar";
import { AccountSettingsContextMenu } from "../context-menu/account-settings-context-menu";
import { useShouldShowUserFeatures } from "#/hooks/use-should-show-user-features";
import { cn } from "#/utils/utils";
import { useConfig } from "#/hooks/query/use-config";

interface UserActionsProps {
  onLogout: () => void;
  user?: { avatar_url: string };
  isLoading?: boolean;
}

export function UserActions({ onLogout, user, isLoading }: UserActionsProps) {
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
    onLogout();
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
            // Base styles: hidden by default with smooth transition
            "opacity-0 pointer-events-none transition-all duration-200",
            // Show on parent hover (avatar)
            "group-hover:opacity-100 group-hover:pointer-events-auto",
            // Keep visible when hovering directly on menu (fixes diagonal cursor movement)
            "hover:opacity-100 hover:pointer-events-auto",
            // Show when menu is explicitly toggled via click
            showMenu && "opacity-100 pointer-events-auto",
          )}
        >
          <AccountSettingsContextMenu
            onLogout={handleLogout}
            onClose={closeAccountMenu}
          />
        </div>
      )}
    </div>
  );
}
