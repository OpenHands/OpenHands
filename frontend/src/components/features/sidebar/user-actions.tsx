import React from "react";
import { UserAvatar } from "./user-avatar";
import { useMe } from "#/hooks/query/use-me";
import { useShouldShowUserFeatures } from "#/hooks/use-should-show-user-features";
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

  // Use the shared hook to determine if user actions should be shown
  const shouldShowUserActions = useShouldShowUserFeatures();

  const showAccountMenu = () => {
    setAccountContextMenuIsVisible(true);
  };

  const hideAccountMenu = () => {
    setAccountContextMenuIsVisible(false);
  };

  const closeAccountMenu = () => {
    if (accountContextMenuIsVisible) {
      setAccountContextMenuIsVisible(false);
    }
  };

  return (
    <div
      data-testid="user-actions"
      className="relative cursor-pointer group"
      onMouseEnter={showAccountMenu}
      onMouseLeave={hideAccountMenu}
    >
      <UserAvatar avatarUrl={user?.avatar_url} isLoading={isLoading} />

      {shouldShowUserActions && user && (
        <div
          className={cn(
            "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto",
            accountContextMenuIsVisible && "opacity-100 pointer-events-auto",
            // Invisible hover bridge: extends hover zone to create a "safe corridor"
            // for diagonal mouse movement to the menu (only active when menu is visible)
            "group-hover:before:content-[''] group-hover:before:block group-hover:before:absolute group-hover:before:inset-[-320px] group-hover:before:z-9998",
          )}
        >
          <UserContextMenu
            type={me?.role || "member"}
            onClose={closeAccountMenu}
          />
        </div>
      )}
    </div>
  );
}
