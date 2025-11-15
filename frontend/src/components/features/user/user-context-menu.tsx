import React from "react";
import ReactDOM from "react-dom";
import { useNavigate } from "react-router";
import {
  IoCardOutline,
  IoLogOutOutline,
  IoPersonAddOutline,
  IoPersonOutline,
} from "react-icons/io5";
import { FaCog } from "react-icons/fa";
import { FaPlus } from "react-icons/fa6";
import { useLogout } from "#/hooks/mutation/use-logout";
import { CreateNewOrganizationModal } from "../org/create-new-organization-modal";
import { OrganizationUserRole } from "#/types/org";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { cn } from "#/utils/utils";
import { InviteOrganizationMemberModal } from "../org/invite-organization-member-modal";
import { useSelectedOrganizationId } from "#/context/use-selected-organization";
import { useOrganizations } from "#/hooks/query/use-organizations";
import { SettingsDropdownInput } from "../settings/settings-dropdown-input";

interface TempButtonProps {
  start: React.ReactNode;
  onClick: () => void;
}

function TempButton({
  start,
  children,
  onClick,
}: React.PropsWithChildren<TempButtonProps>) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1 cursor-pointer hover:text-white w-full"
    >
      {start}
      {children}
    </button>
  );
}

function TempDivider() {
  return <div className="h-[1px] w-full bg-tertiary my-1.5" />;
}

interface UserContextMenuProps {
  type: OrganizationUserRole;
  onClose: () => void;
}

export function UserContextMenu({ type, onClose }: UserContextMenuProps) {
  const navigate = useNavigate();
  const { orgId, setOrgId } = useSelectedOrganizationId();
  const { data: organizations } = useOrganizations();
  const { mutate: logout } = useLogout();
  const ref = useClickOutsideElement<HTMLDivElement>(onClose);

  const [orgModalIsOpen, setOrgModalIsOpen] = React.useState(false);
  const [inviteMemberModalIsOpen, setInviteMemberModalIsOpen] =
    React.useState(false);

  const isUser = type === "user";
  const isSuperAdmin = type === "superadmin";

  const handleLogout = () => {
    logout();
    onClose();
  };

  const handleSettingsClick = () => {
    navigate("/settings");
    onClose();
  };

  const handleInviteMemberClick = () => {
    setInviteMemberModalIsOpen(true);
  };

  const handleManageTeamClick = () => {
    navigate("/settings/team");
    onClose();
  };

  const handleManageAccountClick = () => {
    navigate("/settings/org");
    onClose();
  };

  const handleCreateNewOrgClick = () => {
    setOrgModalIsOpen(true);
  };

  return (
    <div
      data-testid="user-context-menu"
      ref={ref}
      className={cn(
        "w-full flex flex-col gap-3 bg-base border border-tertiary rounded-xl p-6",
        "text-sm text-basic w-fit",
      )}
    >
      {orgModalIsOpen &&
        ReactDOM.createPortal(
          <CreateNewOrganizationModal
            onClose={() => setOrgModalIsOpen(false)}
            onSuccess={() => setInviteMemberModalIsOpen(true)}
          />,
          document.getElementById("portal-root") || document.body,
        )}
      {inviteMemberModalIsOpen &&
        ReactDOM.createPortal(
          <InviteOrganizationMemberModal
            onClose={() => setInviteMemberModalIsOpen(false)}
          />,
          document.getElementById("portal-root") || document.body,
        )}

      <h3 className="text-lg font-semibold text-white">Account</h3>

      <div className="flex flex-col items-start gap-2">
        <SettingsDropdownInput
          testId="org-selector"
          name="organization"
          placeholder="Please select an organization"
          selectedKey={orgId || "personal"}
          items={[
            { key: "personal", label: "Personal Account" },
            ...(organizations?.map((org) => ({
              key: org.id,
              label: org.name,
            })) || [])
          ]}
          onSelectionChange={(org) => {
            if (org === "personal") {
              setOrgId(null);
            } else if (org) {
              setOrgId(org.toString());
            } else {
              setOrgId(null);
            }
          }}
        />

        {!isUser && (
          <>
            <TempButton
              onClick={handleInviteMemberClick}
              start={<IoPersonAddOutline className="text-white" size={14} />}
            >
              Invite Team
            </TempButton>

            <TempDivider />

            <TempButton
              onClick={handleManageAccountClick}
              start={<IoCardOutline className="text-white" size={14} />}
            >
              Manage Account
            </TempButton>
            <TempButton
              onClick={handleManageTeamClick}
              start={<IoPersonOutline className="text-white" size={14} />}
            >
              Manage Team
            </TempButton>
          </>
        )}

        <TempDivider />

        <TempButton
          onClick={handleSettingsClick}
          start={<FaCog className="text-white" size={14} />}
        >
          Settings
        </TempButton>

        {isSuperAdmin && (
          <TempButton
            onClick={handleCreateNewOrgClick}
            start={<FaPlus className="text-white" size={14} />}
          >
            Create New Organization
          </TempButton>
        )}

        <TempButton
          onClick={handleLogout}
          start={<IoLogOutOutline className="text-white" size={14} />}
        >
          Logout
        </TempButton>
      </div>
    </div>
  );
}
