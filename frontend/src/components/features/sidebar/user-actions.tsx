import React from "react";
import ReactDOM from "react-dom";
import { UserAvatar } from "./user-avatar";
import { useMe } from "#/hooks/query/use-me";
import { UserContextMenu } from "../user/user-context-menu";
import { InviteOrganizationMemberModal } from "../org/invite-organization-member-modal";
import { cn } from "#/utils/utils";

interface UserActionsProps {
  user?: { avatar_url: string };
  isLoading?: boolean;
}

export function UserActions({ user, isLoading }: UserActionsProps) {
  const { data: me } = useMe();
  const [accountContextMenuIsVisible, setAccountContextMenuIsVisible] =
    React.useState(false);
  // Counter that increments each time the menu hides, used as a React key
  // to force UserContextMenu to remount with fresh state (resets dropdown
  // open/close, search text, and scroll position in the org selector).
  const [menuResetCount, setMenuResetCount] = React.useState(0);
  const [inviteMemberModalIsOpen, setInviteMemberModalIsOpen] =
    React.useState(false);
  const hideTimeoutRef = React.useRef<number | null>(null);

  // Clean up timeout on unmount
  React.useEffect(
    () => () => {
      if (hideTimeoutRef.current) {
        clearTimeout(hideTimeoutRef.current);
      }
    },
    [],
  );

  const showAccountMenu = () => {
    // Cancel any pending hide to allow diagonal mouse movement to menu
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
    setAccountContextMenuIsVisible(true);
  };

  const hideAccountMenu = () => {
    // Delay hiding to allow diagonal mouse movement to menu
    hideTimeoutRef.current = window.setTimeout(() => {
      setAccountContextMenuIsVisible(false);
      setMenuResetCount((c) => c + 1);
    }, 500);
  };

  const closeAccountMenu = () => {
    if (hideTimeoutRef.current) {
      clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = null;
    }
    if (accountContextMenuIsVisible) {
      setAccountContextMenuIsVisible(false);
      setMenuResetCount((c) => c + 1);
    }
  };

  const openInviteMemberModal = () => {
    setInviteMemberModalIsOpen(true);
  };

  return (
    <>
      <div
        data-testid="user-actions"
        className="relative cursor-pointer group"
        onMouseEnter={showAccountMenu}
        onMouseLeave={hideAccountMenu}
      >
        <UserAvatar avatarUrl={user?.avatar_url} isLoading={isLoading} />

        <div
          className={cn(
            "opacity-0 pointer-events-none group-hover:opacity-100 group-hover:pointer-events-auto",
            accountContextMenuIsVisible && "opacity-100 pointer-events-auto",
          )}
        >
          <UserContextMenu
            key={menuResetCount}
            type={me?.role ?? "member"}
            onClose={closeAccountMenu}
            onOpenInviteModal={openInviteMemberModal}
          />
        </div>
      </div>

      {inviteMemberModalIsOpen &&
        ReactDOM.createPortal(
          <InviteOrganizationMemberModal
            onClose={() => setInviteMemberModalIsOpen(false)}
          />,
          document.getElementById("portal-root") || document.body,
        )}
    </>
  );
}
