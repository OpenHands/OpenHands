import React from "react";
import { useTranslation } from "react-i18next";
import { MessageSquare, Plus, UserRound } from "lucide-react";
import type { AgentProfileSummary } from "#/api/agent-profiles-service/agent-profiles-service.api";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { useNavigation } from "#/context/navigation-context";
import {
  ORGANIZATION_DEPARTMENTS,
  ORGANIZATION_ROLES,
  organizationProfileName,
  parseOrganizationProfileName,
  type OrganizationDepartment,
  type OrganizationDepartmentId,
  type OrganizationRole,
} from "#/constants/agent-organization";
import { useClickOutsideElement } from "#/hooks/use-click-outside-element";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { useActivateAgentProfile } from "#/hooks/mutation/use-activate-agent-profile";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { useAgentProfiles } from "#/hooks/query/use-agent-profiles";
import { I18nKey } from "#/i18n/declaration";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { cn } from "#/utils/utils";

const NEW_CHAT_PATH = "/conversations";

interface OrganizationMember {
  profile: AgentProfileSummary;
  /** Template role when the name matches one; drives the localized title. */
  template: OrganizationRole | undefined;
  /** Role segment of the name, shown when there is no template. */
  roleName: string;
}

function groupMembers(profiles: AgentProfileSummary[]) {
  const byDepartment = new Map<OrganizationDepartmentId, OrganizationMember[]>(
    ORGANIZATION_DEPARTMENTS.map((department) => [department.id, []]),
  );
  const unassigned: OrganizationMember[] = [];
  for (const profile of profiles) {
    const parsed = parseOrganizationProfileName(profile.name);
    if (!parsed) {
      unassigned.push({ profile, template: undefined, roleName: profile.name });
    } else {
      byDepartment.get(parsed.department)!.push({
        profile,
        template: ORGANIZATION_ROLES.find(
          (role) => organizationProfileName(role) === profile.name,
        ),
        roleName: parsed.role,
      });
    }
  }
  return { byDepartment, unassigned };
}

export function AgentOrganizationChart() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const { data, isLoading, isError } = useAgentProfiles();
  const saveProfile = useSaveAgentProfile();
  const activateProfile = useActivateAgentProfile();

  if (isLoading) {
    return (
      <div className="flex justify-center py-16">
        <LoadingSpinner size="large" />
      </div>
    );
  }
  if (isError || !data) {
    return (
      <p role="alert" className="py-8 text-sm text-feedback-error">
        {t(I18nKey.SETTINGS$PROFILES_LOAD_ERROR)}
      </p>
    );
  }

  const { profiles, active_agent_profile_id: activeId } = data;
  const { byDepartment, unassigned } = groupMembers(profiles);
  const existingNames = new Set(profiles.map((profile) => profile.name));
  // New roles reuse the model the user already runs with.
  const llmProfileRef =
    profiles.find((profile) => profile.id === activeId)?.llm_profile_ref ??
    profiles.find((profile) => profile.llm_profile_ref)?.llm_profile_ref ??
    null;

  const handleAddRole = async (role: OrganizationRole) => {
    const roleTitle = t(role.titleKey);
    if (!llmProfileRef) {
      displayErrorToast(t(I18nKey.SETTINGS$AGENT_PROFILE_NO_LLM));
      return;
    }
    try {
      await saveProfile.mutateAsync({
        name: organizationProfileName(role),
        profile: {
          agent_kind: "openhands",
          llm_profile_ref: llmProfileRef,
          system_message_suffix: role.instructions,
        },
      });
      displaySuccessToast(t(I18nKey.ORG$ROLE_CREATED, { role: roleTitle }));
    } catch {
      displayErrorToast(t(I18nKey.ORG$ROLE_CREATE_FAILED, { role: roleTitle }));
    }
  };

  const handleChat = async (profileId: string) => {
    if (profileId !== activeId) await activateProfile.mutateAsync(profileId);
    navigate(NEW_CHAT_PATH);
  };

  const renderDepartment = (department: OrganizationDepartment) => (
    <DepartmentSection
      key={department.id}
      title={t(department.titleKey)}
      description={t(department.descriptionKey)}
      members={byDepartment.get(department.id) ?? []}
      activeId={activeId}
      onChat={handleChat}
      availableRoles={ORGANIZATION_ROLES.filter(
        (role) =>
          role.department === department.id &&
          !existingNames.has(organizationProfileName(role)),
      )}
      onAddRole={handleAddRole}
      isAdding={saveProfile.isPending}
    />
  );

  const [leadership, ...teams] = ORGANIZATION_DEPARTMENTS;

  return (
    <div className="flex flex-col gap-6">
      <div className="mx-auto w-full max-w-md">
        {renderDepartment(leadership)}
      </div>
      {/* Reporting line from leadership to the teams. */}
      <div aria-hidden className="relative -my-3 hidden h-6 md:block">
        <span className="absolute left-1/2 top-0 h-3 w-px bg-border" />
        <span className="absolute left-1/4 right-1/4 top-3 h-px bg-border" />
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {teams.map(renderDepartment)}
      </div>
      {unassigned.length > 0 && (
        <DepartmentSection
          title={t(I18nKey.ORG$UNASSIGNED)}
          description={t(I18nKey.ORG$UNASSIGNED_DESC)}
          members={unassigned}
          activeId={activeId}
          onChat={handleChat}
          muted
        />
      )}
    </div>
  );
}

interface DepartmentSectionProps {
  title: string;
  description: string;
  members: OrganizationMember[];
  activeId: string | null;
  onChat: (profileId: string) => void;
  availableRoles?: OrganizationRole[];
  onAddRole?: (role: OrganizationRole) => void;
  isAdding?: boolean;
  muted?: boolean;
}

function DepartmentSection({
  title,
  description,
  members,
  activeId,
  onChat,
  availableRoles = [],
  onAddRole,
  isAdding = false,
  muted = false,
}: DepartmentSectionProps) {
  const { t } = useTranslation("openhands");
  const headingId = React.useId();

  return (
    <section
      aria-labelledby={headingId}
      className={cn(
        "flex flex-col gap-3 rounded-xl border border-border p-4",
        muted ? "border-dashed" : "bg-surface-raised/40",
      )}
    >
      <header className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 id={headingId} className="text-sm font-medium text-contrast">
            {title}
          </h2>
          <p className="mt-0.5 text-xs leading-4 text-text-secondary">
            {description}
          </p>
        </div>
        {onAddRole && availableRoles.length > 0 && (
          <AddRoleMenu
            roles={availableRoles}
            onSelect={onAddRole}
            disabled={isAdding}
          />
        )}
      </header>
      {members.length === 0 ? (
        <p className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-xs text-text-dim">
          {t(I18nKey.ORG$EMPTY_DEPARTMENT)}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {members.map((member) => (
            <AgentCard
              key={member.profile.name}
              member={member}
              isActive={member.profile.id === activeId}
              onChat={onChat}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function AgentCard({
  member,
  isActive,
  onChat,
}: {
  member: OrganizationMember;
  isActive: boolean;
  onChat: (profileId: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const { profile, template } = member;
  const title = template ? t(template.titleKey) : member.roleName;
  const subtitle =
    profile.agent_kind === "acp"
      ? t(I18nKey.SETTINGS$AGENT_TYPE_ACP)
      : profile.llm_profile_ref;

  return (
    <li className="flex items-center gap-3 rounded-lg border border-border bg-surface px-3 py-2.5">
      <span
        aria-hidden
        className="flex size-8 shrink-0 items-center justify-center rounded-full bg-tertiary text-muted"
      >
        <UserRound className="size-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-sm text-contrast">{title}</span>
          {isActive && (
            <span className="shrink-0 rounded-full bg-feedback-success/15 px-1.5 py-0.5 text-xs font-medium text-feedback-success">
              {t(I18nKey.SETTINGS$PROFILE_ACTIVE)}
            </span>
          )}
        </div>
        {subtitle && (
          <p className="truncate text-xs text-text-dim">{subtitle}</p>
        )}
      </div>
      {profile.id && (
        <button
          type="button"
          aria-label={t(I18nKey.ORG$CHAT_WITH, { name: title })}
          onClick={() => onChat(profile.id!)}
          className="flex shrink-0 cursor-pointer items-center gap-1 rounded-md border border-border px-2 py-1 text-xs text-contrast hover:bg-surface-raised"
        >
          <MessageSquare className="size-3.5" aria-hidden />
          {t(I18nKey.ORG$CHAT)}
        </button>
      )}
    </li>
  );
}

function AddRoleMenu({
  roles,
  onSelect,
  disabled,
}: {
  roles: OrganizationRole[];
  onSelect: (role: OrganizationRole) => void;
  disabled: boolean;
}) {
  const { t } = useTranslation("openhands");
  const [isOpen, setIsOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const close = React.useCallback(() => setIsOpen(false), []);
  const menuRef = useClickOutsideElement<HTMLDivElement>(close, triggerRef);
  useCloseOnEscape(isOpen, close, triggerRef);

  return (
    <div className="relative shrink-0">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        disabled={disabled}
        onClick={() => setIsOpen((open) => !open)}
        className="flex cursor-pointer items-center gap-1 rounded-md px-2 py-1 text-xs text-text-secondary hover:bg-surface-raised hover:text-contrast disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Plus className="size-3.5" aria-hidden />
        {t(I18nKey.ORG$ADD_ROLE)}
      </button>
      {isOpen && (
        <div
          ref={menuRef}
          role="menu"
          className="absolute right-0 top-full z-20 mt-1 flex w-64 flex-col gap-0.5 rounded-lg border border-border bg-surface p-1 shadow-lg"
        >
          {roles.map((role) => (
            <button
              key={role.id}
              type="button"
              role="menuitem"
              onClick={() => {
                close();
                onSelect(role);
              }}
              className="flex cursor-pointer flex-col items-start rounded-md px-2.5 py-2 text-left hover:bg-interactive-hover"
            >
              <span className="text-sm text-contrast">{t(role.titleKey)}</span>
              <span className="text-xs leading-4 text-text-dim">
                {t(role.descriptionKey)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
