import React from "react";
import { useTranslation } from "react-i18next";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { MarketplaceModal } from "#/components/features/settings/marketplace-modal";
import { DeleteConfirmationModal } from "#/components/features/settings/delete-confirmation-modal";
import { cn } from "#/utils/utils";
import { useSettings } from "#/hooks/query/use-settings";
import {
  SETTINGS_QUERY_KEYS,
  ORGANIZATION_APP_SETTINGS_KEY,
} from "#/hooks/query/query-keys";
import { useSkills } from "#/hooks/query/use-skills";
import { useMe } from "#/hooks/query/use-me";
import { useMarketplaceSkills } from "#/hooks/mutation/use-get-marketplace-skills";
import {
  MarketplaceRegistration,
  MarketplaceWithScope,
  SkillWithState,
} from "#/types/settings";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { I18nKey } from "#/i18n/declaration";
import { organizationService } from "#/api/organization-service/organization-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import SkillsService from "#/api/skills-service";
import EditIcon from "#/icons/u-edit.svg?react";
import DeleteIcon from "#/icons/u-delete.svg?react";

function WhiteToggle({
  isToggled,
  onClick,
  "aria-label": ariaLabel,
  disabled,
  title,
}: {
  isToggled: boolean;
  onClick?: () => void;
  "aria-label"?: string;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      className={cn("cursor-pointer", disabled && "cursor-not-allowed")}
      aria-label={ariaLabel}
      disabled={disabled}
      title={title}
    >
      <div
        className={cn(
          "w-12 h-6 rounded-xl flex items-center p-1.5",
          isToggled && "justify-end bg-white",
          !isToggled && "justify-start bg-base-secondary",
          disabled && "opacity-50",
        )}
      >
        <div
          className={cn(
            "w-3 h-3 rounded-xl",
            isToggled ? "bg-[#0D0F11]" : "bg-tertiary-light",
          )}
        />
      </div>
    </button>
  );
}

function ScopeBadge({ scope }: { scope: "instance" | "org" | "personal" }) {
  const { t } = useTranslation();
  const label = {
    instance: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_INSTANCE),
    org: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_ORG),
    personal: t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_PERSONAL),
  }[scope];

  return (
    <span
      className={cn(
        "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium",
        scope === "instance" && "bg-tertiary text-tertiary-alt",
        scope === "org" && "bg-blue-900/30 text-blue-400",
        scope === "personal" && "bg-green-900/30 text-green-400",
      )}
    >
      {label}
    </span>
  );
}

function SkillsSettingsScreen() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();

  // User data for permission checks
  const { data: user } = useMe();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  // Fetch org app settings with updated_at for optimistic locking
  const { data: orgAppSettings } = useQuery({
    queryKey: ORGANIZATION_APP_SETTINGS_KEY,
    queryFn: () => organizationService.getOrganizationAppSettings(),
    retry: false,
  });

  // Determine user role and permissions
  const userRole = user?.role ?? "member";
  const isAdminOrOwner = userRole === "admin" || userRole === "owner";

  // Skills state with marketplace information
  const [skillsState, setSkillsState] = React.useState<SkillWithState[]>([]);
  const [hasSkillChanges, setHasSkillChanges] = React.useState(false);

  // All marketplaces (instance + org + personal) for display
  const [allMarketplaces, setAllMarketplaces] = React.useState<
    MarketplaceWithScope[]
  >([]);
  // Personal marketplaces only (for persistence)
  const [personalMarketplaces, setPersonalMarketplaces] = React.useState<
    MarketplaceRegistration[]
  >([]);
  // Org marketplaces only (for persistence)
  const [orgMarketplaces, setOrgMarketplaces] = React.useState<
    MarketplaceRegistration[]
  >([]);
  // Marketplace changes tracking (auto_load toggles in table)
  const [hasMarketplaceChanges, setHasMarketplaceChanges] =
    React.useState(false);

  // Track last_known_updated_at for optimistic locking
  const [lastKnownUpdatedAt, setLastKnownUpdatedAt] = React.useState<
    string | null
  >(null);

  // Marketplace modal state
  const [isModalOpen, setIsModalOpen] = React.useState(false);
  const [modalMode, setModalMode] = React.useState<"add" | "edit">("add");
  const [selectedMarketplace, setSelectedMarketplace] =
    React.useState<MarketplaceWithScope | null>(null);
  const [selectedScope, setSelectedScope] = React.useState<"org" | "personal">(
    "personal",
  );

  // Delete confirmation modal state
  const [isDeleteModalOpen, setIsDeleteModalOpen] = React.useState(false);
  const [marketplaceToDelete, setMarketplaceToDelete] =
    React.useState<MarketplaceWithScope | null>(null);

  // Save mutations
  const [isSavingPersonal, setIsSavingPersonal] = React.useState(false);
  const [isSavingOrg, setIsSavingOrg] = React.useState(false);
  const [isDeleting, setIsDeleting] = React.useState(false);

  // Marketplace skills mutation for validation
  const marketplaceSkillsMutation = useMarketplaceSkills();

  // Skills filters
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedType, setSelectedType] = React.useState<string | null>(null);
  const [selectedRepository, setSelectedRepository] = React.useState<
    string | null
  >(null);

  React.useEffect(() => {
    if (orgAppSettings?.updated_at) {
      setLastKnownUpdatedAt(orgAppSettings.updated_at);
    }
  }, [orgAppSettings?.updated_at]);

  React.useEffect(() => {
    if (settings && skills) {
      // Build marketplace maps
      const personalMap = new Map(
        (settings.registered_marketplaces || []).map((mp) => [mp.source, mp]),
      );
      const orgMap = new Map(
        (orgAppSettings?.registered_marketplaces || []).map((mp) => [
          mp.source,
          mp,
        ]),
      );
      const instanceMap = new Map(
        (settings.inherited_marketplaces || [])
          .filter((mp) => mp.scope === "instance")
          .map((mp) => [mp.source, mp]),
      );

      // Build all marketplaces for display (deduplicated by source)
      // Use Map to ensure no duplicates - later scopes override earlier ones
      const allBySource = new Map<string, MarketplaceWithScope>();
      // Instance marketplaces (read-only, lowest priority)
      for (const mp of settings.inherited_marketplaces || []) {
        if (mp.scope === "instance") {
          allBySource.set(mp.source, { ...mp, scope: "instance" });
        }
      }
      // Org marketplaces (override instance)
      for (const mp of orgAppSettings?.registered_marketplaces || []) {
        allBySource.set(mp.source, { ...mp, scope: "org" });
      }
      // Personal marketplaces (override org and instance)
      for (const mp of settings.registered_marketplaces || []) {
        allBySource.set(mp.source, { ...mp, scope: "personal" });
      }
      const all = Array.from(allBySource.values());
      setAllMarketplaces(all);
      setPersonalMarketplaces(settings.registered_marketplaces || []);
      setOrgMarketplaces(orgAppSettings?.registered_marketplaces || []);

      // Build combined marketplace lookup for skills
      const marketplaceMap = new Map<string, MarketplaceRegistration>();
      for (const [source, mp] of personalMap) {
        marketplaceMap.set(source, mp);
      }
      for (const [source, mp] of orgMap) {
        marketplaceMap.set(source, mp);
      }
      for (const [source, mp] of instanceMap) {
        marketplaceMap.set(source, mp);
      }

      // Map skills with marketplace info
      const disabledSet = new Set(settings.disabled_skills || []);
      const mappedSkills: SkillWithState[] = skills.map((skill) => {
        let repoUrl = skill.source;
        let skillScope: "instance" | "org" | "personal" = "personal";

        if (skill.source === "global") {
          skillScope = "instance";
        } else if (skill.source === "user") {
          skillScope = "personal";
        } else {
          // Check which marketplace this skill belongs to
          const marketplace =
            marketplaceMap.get(skill.name) || marketplaceMap.get(skill.source);
          if (marketplace) {
            repoUrl = marketplace.source;
            // Find the scope from inherited marketplaces or determine from registry
            const inheritedMp = (settings.inherited_marketplaces || []).find(
              (mp) =>
                mp.source === marketplace.source &&
                (mp as MarketplaceWithScope).scope === "instance",
            );
            if (inheritedMp) {
              skillScope = "instance";
            } else {
              // Check org vs personal
              const orgMp = (
                orgAppSettings?.registered_marketplaces || []
              ).find((mp) => mp.source === marketplace.source);
              if (orgMp) {
                skillScope = "org";
              } else {
                skillScope = "personal";
              }
            }
          }
        }

        return {
          ...skill,
          id: skill.name,
          repository: repoUrl,
          scope: skillScope,
          isEnabled: !disabledSet.has(skill.name),
          isAutoLoad: marketplaceMap.get(skill.name)?.auto_load === "all",
        };
      });

      // Fetch marketplace skills and merge with global/user skills
      const fetchMarketplaceSkills = async () => {
        const allRegisteredMarketplaces = [
          ...(settings.registered_marketplaces || []),
          ...(orgAppSettings?.registered_marketplaces || []),
        ];

        if (allRegisteredMarketplaces.length === 0) {
          setSkillsState(mappedSkills);
          return;
        }

        try {
          const preview = await SkillsService.getMarketplaceSkills(
            allRegisteredMarketplaces,
          );

          // Show errors if any
          if (preview.errors && preview.errors.length > 0) {
            preview.errors.forEach((error) => {
              displayErrorToast(`Marketplace error: ${error}`);
            });
          }

          if (preview.skills.length > 0) {
            // Deduplicate skills by name before adding - prevents duplicates on refresh
            const seenSkillNames = new Set<string>();
            const marketplaceSkills: SkillWithState[] = [];

            for (const skill of preview.skills) {
              if (!seenSkillNames.has(skill.name)) {
                seenSkillNames.add(skill.name);

                // Determine scope based on which marketplace this came from
                const marketplace = allRegisteredMarketplaces.find(
                  (mp) => skill.source === `marketplace:${mp.name}`,
                );
                // Use SOURCE-based matching instead of NAME-based to correctly determine scope
                const isOrg = marketplace
                  ? (orgAppSettings?.registered_marketplaces || []).some(
                      (mp) => mp.source === marketplace.source,
                    )
                  : false;

                marketplaceSkills.push({
                  ...skill,
                  id: skill.name,
                  repository: marketplace?.source || skill.source,
                  scope: isOrg ? "org" : "personal",
                  isEnabled: !disabledSet.has(skill.name),
                  isAutoLoad: marketplace?.auto_load === "all",
                });
              }
            }

            // Merge global/user skills with marketplace skills (both already deduplicated)
            setSkillsState([...mappedSkills, ...marketplaceSkills]);
          } else {
            setSkillsState(mappedSkills);
          }
        } catch (error) {
          // Fall back to just global/user skills
          setSkillsState(mappedSkills);
        }
      };

      fetchMarketplaceSkills();
    }
  }, [settings, skills, orgAppSettings]);

  const filteredSkills = React.useMemo(
    () =>
      skillsState.filter((skill) => {
        const matchesSearch =
          !searchQuery ||
          skill.name.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesType =
          !selectedType ||
          selectedType === "all" ||
          skill.type.toLowerCase() === selectedType.toLowerCase();
        const matchesRepo =
          !selectedRepository ||
          selectedRepository === "all" ||
          skill.repository === selectedRepository;
        return matchesSearch && matchesType && matchesRepo;
      }),
    [skillsState, searchQuery, selectedType, selectedRepository],
  );

  const typeOptions = React.useMemo(() => {
    const types = new Set(skillsState.map((s) => s.type));
    return [
      { key: "all", label: t(I18nKey.SETTINGS$ALL_TYPES) },
      ...Array.from(types).map((type) => ({
        key: type.toLowerCase(),
        label: type.charAt(0).toUpperCase() + type.slice(1),
      })),
    ];
  }, [skillsState, t]);

  const repositoryOptions = React.useMemo(() => {
    const repos = new Set(skillsState.map((s) => s.repository));
    return [
      { key: "all", label: t(I18nKey.SETTINGS$ALL_REPOSITORIES) },
      ...Array.from(repos).map((repo) => ({
        key: repo,
        label: repo,
      })),
    ];
  }, [skillsState, t]);

  // Toggle handlers with permission checks
  const handleToggleEnabled = (skillId: string) => {
    const skill = skillsState.find((s) => s.id === skillId);
    if (!skill) return;

    // Permission check: Org requires admin/owner, Personal requires active user
    const canToggle =
      skill.scope === "instance" ||
      (skill.scope === "org" && isAdminOrOwner) ||
      skill.scope === "personal";

    if (!canToggle) return;

    setSkillsState((prev) =>
      prev.map((s) =>
        s.id === skillId ? { ...s, isEnabled: !s.isEnabled } : s,
      ),
    );
    setHasSkillChanges(true);
  };

  // Save skill changes (disabled_skills)
  const handleSaveSkillChanges = async () => {
    const disabledSkills = skillsState
      .filter((s) => !s.isEnabled)
      .map((s) => s.name);

    setIsSavingPersonal(true);
    try {
      await SettingsService.saveSettings({ disabled_skills: disabledSkills });
      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      setHasSkillChanges(false);
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
    } catch (error) {
      const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
      displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
    } finally {
      setIsSavingPersonal(false);
    }
  };

  // Save marketplace changes (auto_load toggles)
  const handleSaveMarketplaceChanges = async () => {
    setIsSavingPersonal(true);
    setIsSavingOrg(true);

    try {
      // Save personal marketplaces
      await SettingsService.saveSettings({
        registered_marketplaces: personalMarketplaces,
      });

      // Save org marketplaces
      await organizationService.saveOrganizationAppSettings({
        registered_marketplaces: orgMarketplaces,
        last_known_updated_at: lastKnownUpdatedAt,
      });

      displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
      setHasMarketplaceChanges(false);
      queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
      queryClient.invalidateQueries({
        queryKey: ORGANIZATION_APP_SETTINGS_KEY,
      });
    } catch (error) {
      const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
      displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
    } finally {
      setIsSavingPersonal(false);
      setIsSavingOrg(false);
    }
  };

  // Marketplace modal handlers
  const openAddModal = (scope: "org" | "personal" = "personal") => {
    setModalMode("add");
    setSelectedMarketplace(null);
    setSelectedScope(scope);
    setIsModalOpen(true);
  };

  const openEditModal = (marketplace: MarketplaceWithScope) => {
    setModalMode("edit");
    setSelectedMarketplace(marketplace);
    setIsModalOpen(true);
  };

  const openDeleteModal = (marketplace: MarketplaceWithScope) => {
    setMarketplaceToDelete(marketplace);
    setIsDeleteModalOpen(true);
  };

  const handleSaveMarketplace = async (data: {
    name: string;
    source: string;
    ref?: string;
    repo_path?: string;
    auto_load?: "all";
    scope: "org" | "personal";
  }) => {
    const newMarketplace: MarketplaceRegistration = {
      name: data.name,
      source: data.source,
      ref: data.ref,
      repo_path: data.repo_path,
      auto_load: data.auto_load,
    };

    if (data.scope === "org") {
      // Save to org settings
      setIsSavingOrg(true);
      const existingIndex = orgMarketplaces.findIndex(
        (mp) => mp.source === data.source,
      );
      let updated: MarketplaceRegistration[];
      if (existingIndex >= 0) {
        updated = [...orgMarketplaces];
        updated[existingIndex] = newMarketplace;
      } else {
        updated = [...orgMarketplaces, newMarketplace];
      }

      try {
        // Validate marketplace by fetching skills first
        const preview = await marketplaceSkillsMutation.mutateAsync([
          newMarketplace,
        ]);

        // Check for errors - don't save if validation fails
        if (preview.errors && preview.errors.length > 0) {
          displayErrorToast(
            `Failed to validate marketplace: ${preview.errors.join(", ")}`,
          );
          setIsSavingOrg(false);
          return;
        }

        // Only save to BE if validation passes
        await organizationService.saveOrganizationAppSettings({
          registered_marketplaces: updated,
          last_known_updated_at: lastKnownUpdatedAt,
        });
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        setOrgMarketplaces(updated);
        queryClient.invalidateQueries({
          queryKey: ORGANIZATION_APP_SETTINGS_KEY,
        });

        // Skills will be loaded by the useEffect on next render
        // This prevents duplicate skills when page is refreshed

        setIsModalOpen(false);
      } catch (error) {
        if ((error as AxiosError).response?.status === 409) {
          displayErrorToast(
            "Your settings are outdated. Please refresh and try again.",
          );
          queryClient.invalidateQueries({
            queryKey: ORGANIZATION_APP_SETTINGS_KEY,
          });
        } else {
          const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        }
      } finally {
        setIsSavingOrg(false);
      }
    } else {
      // Save to personal settings
      setIsSavingPersonal(true);
      const existingIndex = personalMarketplaces.findIndex(
        (mp) => mp.source === data.source,
      );
      let updated: MarketplaceRegistration[];
      if (existingIndex >= 0) {
        updated = [...personalMarketplaces];
        updated[existingIndex] = newMarketplace;
      } else {
        updated = [...personalMarketplaces, newMarketplace];
      }

      try {
        // Validate marketplace by fetching skills first
        const preview = await marketplaceSkillsMutation.mutateAsync([
          newMarketplace,
        ]);

        // Check for errors - don't save if validation fails
        if (preview.errors && preview.errors.length > 0) {
          displayErrorToast(
            `Failed to validate marketplace: ${preview.errors.join(", ")}`,
          );
          setIsSavingPersonal(false);
          return;
        }

        // Only save to BE if validation passes
        await SettingsService.saveSettings({
          registered_marketplaces: updated,
        });
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        setPersonalMarketplaces(updated);
        queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });

        // Skills will be loaded by the useEffect on next render
        // This prevents duplicate skills when page is refreshed

        setIsModalOpen(false);
      } catch (error) {
        const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
        displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
      } finally {
        setIsSavingPersonal(false);
      }
    }
  };

  const handleDeleteMarketplace = async () => {
    if (!marketplaceToDelete) return;

    setIsDeleting(true);

    if (marketplaceToDelete.scope === "org") {
      // Delete from org settings
      const updated = orgMarketplaces.filter(
        (mp) => mp.source !== marketplaceToDelete.source,
      );

      try {
        await organizationService.saveOrganizationAppSettings({
          registered_marketplaces: updated,
          last_known_updated_at: lastKnownUpdatedAt,
        });
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        setOrgMarketplaces(updated);

        // Update allMarketplaces immediately to remove the deleted org marketplace
        setAllMarketplaces((prev) =>
          prev.filter((mp) => mp.source !== marketplaceToDelete.source),
        );

        // Remove skills associated with this org marketplace
        setSkillsState((prev) =>
          prev.filter(
            (skill) =>
              !(
                skill.repository === marketplaceToDelete.source &&
                skill.scope === "org"
              ),
          ),
        );

        queryClient.invalidateQueries({
          queryKey: ORGANIZATION_APP_SETTINGS_KEY,
        });
        setIsDeleteModalOpen(false);
        setMarketplaceToDelete(null);
      } catch (error) {
        if ((error as AxiosError).response?.status === 409) {
          displayErrorToast(
            "Your settings are outdated. Please refresh and try again.",
          );
          queryClient.invalidateQueries({
            queryKey: ORGANIZATION_APP_SETTINGS_KEY,
          });
        } else {
          const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        }
        setIsDeleting(false);
      }
    } else if (marketplaceToDelete.scope === "personal") {
      // Delete from personal settings
      const updated = personalMarketplaces.filter(
        (mp) => mp.source !== marketplaceToDelete.source,
      );

      try {
        await SettingsService.saveSettings({
          registered_marketplaces: updated,
        });
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
        setPersonalMarketplaces(updated);

        // Update allMarketplaces immediately to remove the deleted personal marketplace
        setAllMarketplaces((prev) =>
          prev.filter((mp) => mp.source !== marketplaceToDelete.source),
        );

        // Remove skills associated with this personal marketplace
        setSkillsState((prev) =>
          prev.filter(
            (skill) =>
              !(
                skill.repository === marketplaceToDelete.source &&
                skill.scope === "personal"
              ),
          ),
        );

        queryClient.invalidateQueries({ queryKey: SETTINGS_QUERY_KEYS.all });
        setIsDeleteModalOpen(false);
        setMarketplaceToDelete(null);
      } catch (error) {
        const errorMessage = retrieveAxiosErrorMessage(error as AxiosError);
        displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        setIsDeleting(false);
      }
    } else {
      // Instance - should not be deletable
      setIsDeleting(false);
    }
  };

  // Check if user can edit/delete a marketplace
  const canEditMarketplace = (mp: MarketplaceWithScope) => {
    if (mp.scope === "instance") return false; // Always disabled
    if (mp.scope === "org") return isAdminOrOwner;
    if (mp.scope === "personal") return true; // Active user owns their personal
    return false;
  };

  // Get tooltip for disabled actions
  const getDisabledTooltip = (mp: MarketplaceWithScope) => {
    if (mp.scope === "instance") {
      return t(I18nKey.SETTINGS$MARKETPLACE_INSTANCE_READONLY);
    }
    if (mp.scope === "org" && !isAdminOrOwner) {
      return t(I18nKey.SETTINGS$MARKETPLACE_ORG_REQUIRES_ADMIN);
    }
    return "";
  };

  // Get title for auto_load toggle
  const getAutoLoadToggleTitle = (scope: "instance" | "org" | "personal") => {
    if (scope === "instance") {
      return t(I18nKey.SETTINGS$MARKETPLACE_INSTANCE_READONLY);
    }
    if (scope === "org" && !isAdminOrOwner) {
      return t(I18nKey.SETTINGS$MARKETPLACE_ORG_REQUIRES_ADMIN);
    }
    return undefined;
  };

  // Handle marketplace auto_load toggle (persists via Save Changes button)
  const handleToggleMarketplaceAutoLoad = (source: string) => {
    // Determine scope and update appropriate list
    const isOrg = orgMarketplaces.some((mp) => mp.source === source);
    const isPersonal = personalMarketplaces.some((mp) => mp.source === source);

    // Create new state for comparison
    let newOrgMarketplaces = orgMarketplaces;
    let newPersonalMarketplaces = personalMarketplaces;

    if (isOrg) {
      newOrgMarketplaces = orgMarketplaces.map((mp) =>
        mp.source === source
          ? { ...mp, auto_load: mp.auto_load === "all" ? undefined : "all" }
          : mp,
      );
      setOrgMarketplaces(newOrgMarketplaces);
    } else if (isPersonal) {
      newPersonalMarketplaces = personalMarketplaces.map((mp) =>
        mp.source === source
          ? { ...mp, auto_load: mp.auto_load === "all" ? undefined : "all" }
          : mp,
      );
      setPersonalMarketplaces(newPersonalMarketplaces);
    }

    // Check if all marketplace values match their original state
    let hasChanges = false;
    for (const mp of allMarketplaces) {
      if (mp.scope === "personal") {
        const current = newPersonalMarketplaces.find(
          (p) => p.source === mp.source,
        );
        if (current && current.auto_load !== mp.auto_load) {
          hasChanges = true;
          break;
        }
      } else if (mp.scope === "org") {
        const current = newOrgMarketplaces.find((p) => p.source === mp.source);
        if (current && current.auto_load !== mp.auto_load) {
          hasChanges = true;
          break;
        }
      }
      // instance scope doesn't change
    }
    setHasMarketplaceChanges(hasChanges);
  };

  // Check if skill toggles should be disabled
  const isSkillToggleDisabled = (skill: SkillWithState) => {
    if (skill.scope === "org") {
      return !isAdminOrOwner;
    }
    return false; // Instance and Personal are always allowed
  };

  const isLoading = settingsLoading || skillsLoading || !settings;

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <div className="mb-8">
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$MARKETPLACES)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$MARKETPLACES_DESCRIPTION)}
          </Typography.Paragraph>
        </div>
        <div className="flex items-center justify-center h-64">
          <div className="animate-pulse text-content-secondary">Loading...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="mb-8">
        <Typography.H2 className="mb-2">
          {t(I18nKey.SETTINGS$MARKETPLACES)}
        </Typography.H2>
        <Typography.Paragraph className="text-sm text-[#8c8c8c]">
          {t(I18nKey.SETTINGS$MARKETPLACES_DESCRIPTION)}
        </Typography.Paragraph>
      </div>

      {/* Marketplace Table */}
      <section className="mb-8 flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <Typography.H2 className="mb-2">
              {t(I18nKey.SETTINGS$CONNECT_MARKETPLACES)}
            </Typography.H2>
            <BrandButton
              testId="add-marketplace-button"
              variant="primary"
              type="button"
              onClick={() => openAddModal("personal")}
            >
              {t(I18nKey.SETTINGS$MARKETPLACE_ADD)}
            </BrandButton>
          </div>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$CONNECT_MARKETPLACES_DESCRIPTION)}
          </Typography.Paragraph>
        </div>

        <div className="border border-tertiary rounded-md overflow-hidden">
          <table className="w-full">
            <thead className="bg-base-secondary">
              <tr className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr_auto] gap-4 items-center">
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_NAME)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_REF)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_REPO_PATH)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$AUTO_LOAD)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$ACTIONS)}
                </th>
              </tr>
            </thead>
            <tbody>
              {allMarketplaces.map((mp) => (
                <tr
                  key={mp.source}
                  className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr_auto] gap-4 items-center border-t border-tertiary"
                >
                  <td className="p-3 text-sm text-content-2 truncate min-w-0">
                    {mp.name}
                  </td>
                  <td className="p-3 text-sm text-tertiary-alt truncate">
                    {mp.source}
                  </td>
                  <td className="p-3 text-sm text-tertiary-alt truncate">
                    {mp.ref || "-"}
                  </td>
                  <td className="p-3 text-sm text-tertiary-alt truncate">
                    {mp.repo_path || "-"}
                  </td>
                  <td className="p-3">
                    <ScopeBadge scope={mp.scope} />
                  </td>
                  <td className="p-3">
                    <WhiteToggle
                      isToggled={mp.auto_load === "all"}
                      disabled={
                        mp.scope === "instance" ||
                        (mp.scope === "org" && !isAdminOrOwner)
                      }
                      onClick={
                        mp.scope !== "instance" &&
                        (mp.scope === "personal" || isAdminOrOwner)
                          ? () => handleToggleMarketplaceAutoLoad(mp.source)
                          : undefined
                      }
                      title={getAutoLoadToggleTitle(mp.scope)}
                      aria-label={`Toggle auto-load for ${mp.source}`}
                    />
                  </td>
                  <td className="p-3 flex gap-2 justify-center">
                    <button
                      type="button"
                      onClick={() => openEditModal(mp)}
                      disabled={!canEditMarketplace(mp)}
                      title={getDisabledTooltip(mp) || t(I18nKey.BUTTON$EDIT)}
                      className={cn(
                        "p-1.5 rounded-sm",
                        canEditMarketplace(mp)
                          ? "text-content-2 hover:bg-white/20"
                          : "text-tertiary-alt cursor-not-allowed opacity-50",
                      )}
                    >
                      <EditIcon width={16} height={16} />
                    </button>
                    <button
                      type="button"
                      onClick={() => openDeleteModal(mp)}
                      disabled={!canEditMarketplace(mp)}
                      title={getDisabledTooltip(mp) || t(I18nKey.BUTTON$DELETE)}
                      className={cn(
                        "p-1.5 rounded-sm",
                        canEditMarketplace(mp)
                          ? "text-red-400 hover:bg-red-900/20"
                          : "text-tertiary-alt cursor-not-allowed opacity-50",
                      )}
                    >
                      <DeleteIcon width={16} height={16} />
                    </button>
                  </td>
                </tr>
              ))}
              {allMarketplaces.length === 0 && (
                <tr className="border-t border-tertiary">
                  <td
                    colSpan={7}
                    className="p-3 text-sm text-center text-tertiary-alt"
                  >
                    {t(I18nKey.SETTINGS$MARKETPLACE_ADD_FIRST)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="border-t border-tertiary my-6" />

      {/* Skills Table */}
      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$SKILLS)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$SKILLS_DESCRIPTION)}
          </Typography.Paragraph>
        </div>

        <div className="flex items-stretch gap-4 justify-center">
          <div className="flex-1 flex flex-col gap-2.5">
            <div className="h-5" />
            <input
              data-testid="search-skills-input"
              type="text"
              placeholder={t(I18nKey.SETTINGS$SEARCH_PLACEHOLDER)}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt"
            />
          </div>
          <div className="flex-1">
            <SettingsDropdownInput
              testId="type-filter-dropdown"
              name="type-filter"
              label="TYPE"
              items={typeOptions}
              defaultSelectedKey="all"
              onSelectionChange={(key) =>
                setSelectedType(key?.toString() ?? null)
              }
              placeholder={t(I18nKey.SETTINGS$ALL_TYPES)}
            />
          </div>
          <div className="flex-1">
            <SettingsDropdownInput
              testId="repository-filter-dropdown"
              name="repository-filter"
              label="REPOSITORY"
              items={repositoryOptions}
              defaultSelectedKey="all"
              onSelectionChange={(key) =>
                setSelectedRepository(key?.toString() ?? null)
              }
              placeholder={t(I18nKey.SETTINGS$ALL_REPOSITORIES)}
            />
          </div>
        </div>

        <div className="border border-tertiary rounded-md overflow-hidden">
          <table className="w-full">
            <thead className="bg-base-secondary">
              <tr className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] gap-4 items-start">
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$TYPE)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_LABEL)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$ENABLED)}
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredSkills.map((skill) => (
                <tr
                  key={skill.id}
                  className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr] gap-4 items-center border-t border-tertiary"
                >
                  <td className="p-3 text-sm text-content-2 truncate min-w-0">
                    {skill.name}
                  </td>
                  <td className="p-3 text-sm text-tertiary-alt truncate">
                    {skill.repository}
                  </td>
                  <td className="p-3">
                    <span className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-base-secondary text-tertiary-alt">
                      {skill.type}
                    </span>
                  </td>
                  <td className="p-3">
                    <ScopeBadge scope={skill.scope} />
                  </td>
                  <td className="p-3">
                    <WhiteToggle
                      isToggled={skill.isEnabled}
                      disabled={isSkillToggleDisabled(skill)}
                      onClick={() => handleToggleEnabled(skill.id)}
                      title={
                        isSkillToggleDisabled(skill) && skill.scope === "org"
                          ? t(I18nKey.SETTINGS$MARKETPLACE_ORG_REQUIRES_ADMIN)
                          : undefined
                      }
                      aria-label={`Toggle enabled for ${skill.name}`}
                    />
                  </td>
                </tr>
              ))}
              {filteredSkills.length === 0 && (
                <tr className="border-t border-tertiary">
                  <td
                    colSpan={5}
                    className="p-3 text-sm text-center text-[rgb(140,140,140)]"
                  >
                    {t(I18nKey.SETTINGS$NO_SKILLS_MATCH_FILTERS)}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <div className="flex gap-6 p-6 justify-end border-t border-tertiary/50 mt-4">
        <BrandButton
          testId="skills-save-button"
          variant="primary"
          type="button"
          isDisabled={
            isSavingPersonal ||
            isSavingOrg ||
            (!hasSkillChanges && !hasMarketplaceChanges)
          }
          onClick={async () => {
            if (hasSkillChanges) await handleSaveSkillChanges();
            if (hasMarketplaceChanges) await handleSaveMarketplaceChanges();
          }}
        >
          {(isSavingPersonal || isSavingOrg) && t(I18nKey.SETTINGS$SAVING)}
          {!isSavingPersonal &&
            !isSavingOrg &&
            t(I18nKey.SETTINGS$SAVE_CHANGES)}
        </BrandButton>
      </div>

      {/* Marketplace Modal */}
      <MarketplaceModal
        isOpen={isModalOpen}
        mode={modalMode}
        scope={selectedScope}
        marketplace={
          selectedMarketplace
            ? {
                name: selectedMarketplace.name,
                source: selectedMarketplace.source,
                ref: selectedMarketplace.ref,
                repo_path: selectedMarketplace.repo_path,
                auto_load: selectedMarketplace.auto_load,
              }
            : null
        }
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveMarketplace}
        isSaving={isSavingOrg || isSavingPersonal}
        isAdminOrOwner={isAdminOrOwner}
      />

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={isDeleteModalOpen}
        itemName={marketplaceToDelete?.name || ""}
        onClose={() => {
          setIsDeleteModalOpen(false);
          setMarketplaceToDelete(null);
        }}
        onDelete={handleDeleteMarketplace}
        isDeleting={isDeleting}
      />
    </div>
  );
}

export default SkillsSettingsScreen;
