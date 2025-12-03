import React from "react";
import { UserAvatar } from "./user-avatar";
import { useMe } from "#/hooks/query/use-me";
import { useShouldShowUserFeatures } from "#/hooks/use-should-show-user-features";
import { UserContextMenu } from "../user/user-context-menu";

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

      {accountContextMenuIsVisible && !!user && shouldShowUserActions && (
        <UserContextMenu type={me?.role || "user"} onClose={closeAccountMenu} />
      )}
    </div>
  );
}
