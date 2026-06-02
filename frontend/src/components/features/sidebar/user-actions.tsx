import React from "react";
import ReactDOM from "react-dom";
import { UserAvatar } from "./user-avatar";
import { useMe } from "#/hooks/query/use-me";
import { UserContextMenu } from "../user/user-context-menu";
import { InviteOrganizationMemberModal } from "../org/invite-organization-member-modal";

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

  const closeAccountMenu = () => {
    setAccountContextMenuIsVisible(false);
    setMenuResetCount((c) => c + 1);
  };

  const toggleAccountMenu = (e: React.MouseEvent) => {
    // Prevent the click from reaching the document, which would trigger
    // ContextMenuContainer's click-outside handler immediately after opening.
    e.stopPropagation();
    if (accountContextMenuIsVisible) {
      closeAccountMenu();
    } else {
      setAccountContextMenuIsVisible(true);
    }
  };

  const openInviteMemberModal = () => {
    setInviteMemberModalIsOpen(true);
  };

  return (
    <>
      <div
        data-testid="user-actions"
        className="relative cursor-pointer"
        onClick={toggleAccountMenu}
      >
        <UserAvatar avatarUrl={user?.avatar_url} isLoading={isLoading} />

        {accountContextMenuIsVisible && (
          // Prevent menu item clicks from bubbling to the toggle handler above.
          // ContextMenuContainer handles its own click-outside-to-close logic.
          <div onClick={(e) => e.stopPropagation()}>
            <UserContextMenu
              key={menuResetCount}
              type={me?.role ?? "member"}
              onClose={closeAccountMenu}
              onOpenInviteModal={openInviteMemberModal}
            />
          </div>
        )}
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
