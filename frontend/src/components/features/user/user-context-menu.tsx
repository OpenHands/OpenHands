import React from "react";
import ReactDOM from "react-dom";
import { Link, useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import {
  IoCardOutline,
  IoLogOutOutline,
  IoPersonAddOutline,
  IoPersonOutline,
} from "react-icons/io5";
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
import { I18nKey } from "#/i18n/declaration";
import { SAAS_NAV_ITEMS, OSS_NAV_ITEMS } from "#/constants/settings-nav";
import { useConfig } from "#/hooks/query/use-config";
import DocumentIcon from "#/icons/document.svg?react";

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
  return <div className="h-[1px] w-full bg-[#5C5D62] my-1.5" />;
}

interface UserContextMenuProps {
  type: OrganizationUserRole;
  onClose: () => void;
}

export function UserContextMenu({ type, onClose }: UserContextMenuProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { orgId, setOrgId } = useSelectedOrganizationId();
  const { data: organizations } = useOrganizations();
  const { mutate: logout } = useLogout();
  const { data: config } = useConfig();
  const ref = useClickOutsideElement<HTMLDivElement>(onClose);

  const isOss = config?.APP_MODE === "oss";
  // Filter out team/org nav items since they're already handled separately in the menu
  const navItems = (isOss ? OSS_NAV_ITEMS : SAAS_NAV_ITEMS).filter(
    (item) => item.to !== "/settings/team" && item.to !== "/settings/org",
  );

  const [orgModalIsOpen, setOrgModalIsOpen] = React.useState(false);
  const [inviteMemberModalIsOpen, setInviteMemberModalIsOpen] =
    React.useState(false);

  const isUser = type === "user";
  const isSuperAdmin = type === "superadmin";

  const handleLogout = () => {
    logout();
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
        "w-64 flex flex-col gap-3 bg-tertiary border border-tertiary rounded-xl p-6",
        "text-sm absolute left-full bottom-0 z-60",
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

      <h3 className="text-lg font-semibold text-white">
        {t(I18nKey.ORG$ACCOUNT)}
      </h3>

      <div className="flex flex-col items-start gap-2">
        <div className="w-full relative">
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
              })) || []),
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
        </div>

        {!isUser && (
          <>
            <TempButton
              onClick={handleInviteMemberClick}
              start={<IoPersonAddOutline className="text-white" size={14} />}
            >
              {t(I18nKey.ORG$INVITE_TEAM)}
            </TempButton>

            <TempDivider />

            <TempButton
              onClick={handleManageAccountClick}
              start={<IoCardOutline className="text-white" size={14} />}
            >
              {t(I18nKey.ORG$MANAGE_ACCOUNT)}
            </TempButton>
            <TempButton
              onClick={handleManageTeamClick}
              start={<IoPersonOutline className="text-white" size={14} />}
            >
              {t(I18nKey.ORG$MANAGE_TEAM)}
            </TempButton>
          </>
        )}

        <TempDivider />

        {navItems.map((item) => (
          <Link
            key={item.to}
            to={item.to}
            onClick={onClose}
            className="flex items-center gap-1 cursor-pointer hover:text-white w-full"
          >
            {React.cloneElement(item.icon, {
              className: "text-white",
              width: 14,
              height: 14,
            } as React.SVGProps<SVGSVGElement>)}
            {t(item.text)}
          </Link>
        ))}

        <TempDivider />

        <a
          href="https://docs.openhands.dev"
          target="_blank"
          rel="noopener noreferrer"
          onClick={onClose}
          className="flex items-center gap-1 cursor-pointer hover:text-white w-full"
        >
          <DocumentIcon className="text-white" width={14} height={14} />
          {t(I18nKey.SIDEBAR$DOCS)}
        </a>

        {isSuperAdmin && (
          <TempButton
            onClick={handleCreateNewOrgClick}
            start={<FaPlus className="text-white" size={14} />}
          >
            {t(I18nKey.ORG$CREATE_NEW_ORGANIZATION)}
          </TempButton>
        )}

        <TempButton
          onClick={handleLogout}
          start={<IoLogOutOutline className="text-white" size={14} />}
        >
          {t(I18nKey.ACCOUNT_SETTINGS$LOGOUT)}
        </TempButton>
      </div>
    </div>
  );
}
