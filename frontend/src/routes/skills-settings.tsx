import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useRef,
} from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { MarketplaceModal } from "#/components/features/settings/marketplace-modal";
import { DeleteConfirmationModal } from "#/components/features/settings/delete-confirmation-modal";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import { useMe } from "#/hooks/query/use-me";
import { useOrganizationAppSettings } from "#/hooks/query/use-organization-app-settings";
import { useOrganization } from "#/hooks/query/use-organization";
import { useMarketplaceMutations } from "#/hooks/mutation/use-marketplace-mutations";
import { useSkillMutations } from "#/hooks/mutation/use-skill-mutations";
import { MarketplaceTable } from "#/components/features/settings/skills-settings/marketplace-table";
import { SkillsTable } from "#/components/features/settings/skills-settings/skills-table";
import { MarketplaceRegistration, SkillWithState } from "#/types/settings";
import { I18nKey } from "#/i18n/declaration";
import SkillsService from "#/api/skills-service";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

function SkillsSettingsScreen() {
  const { t } = useTranslation();

  // Query data
  const { data: user } = useMe();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();
  const { data: orgAppSettings } = useOrganizationAppSettings();
  const { data: currentOrganization } = useOrganization();

  // Permissions
  const userRole = user?.role ?? "member";
  const isAdminOrOwner = userRole === "admin" || userRole === "owner";

  // Active scope derived from current organization
  const activeScope = useMemo((): "org" | "personal" => {
    if (!currentOrganization || currentOrganization.is_personal) {
      return "personal";
    }
    return "org";
  }, [currentOrganization]);

  // State for skills
  const [skillsState, setSkillsState] = useState<SkillWithState[]>([]);
  const originalSkillsRef = useRef<SkillWithState[]>([]);

  // State for marketplaces
  const [allMarketplaces, setAllMarketplaces] = useState<
    MarketplaceRegistration[]
  >([]);
  const originalMarketplacesRef = useRef<MarketplaceRegistration[]>([]);
  const [lastKnownUpdatedAt, setLastKnownUpdatedAt] = useState<string | null>(
    null,
  );

  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<"add" | "edit">("add");
  const [selectedMarketplace, setSelectedMarketplace] =
    useState<MarketplaceRegistration | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [marketplaceToDelete, setMarketplaceToDelete] =
    useState<MarketplaceRegistration | null>(null);

  // Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [selectedRepository, setSelectedRepository] = useState<string | null>(
    null,
  );

  // Mutations from hooks
  const marketplaceMutations = useMarketplaceMutations();
  const skillMutations = useSkillMutations();

  // Derive saving state from mutations
  const isSaving =
    marketplaceMutations.savePersonal.isPending ||
    marketplaceMutations.saveOrg.isPending ||
    skillMutations.saveDisabledSkills.isPending;

  const isDeleting =
    marketplaceMutations.deletePersonal.isPending ||
    marketplaceMutations.deleteOrg.isPending;

  // Change detection
  const hasSkillChanges = useMemo(() => {
    const original = originalSkillsRef.current;
    if (skillsState.length !== original.length) return false;
    const originalById = new Map(original.map((s) => [s.id, s]));
    return skillsState.some((skill) => {
      const orig = originalById.get(skill.id);
      return orig?.isEnabled !== skill.isEnabled;
    });
  }, [skillsState]);

  const hasMarketplaceChanges = useMemo(() => {
    const original = originalMarketplacesRef.current;
    if (allMarketplaces.length !== original.length) return false;
    const originalBySource = new Map(original.map((mp) => [mp.source, mp]));
    return allMarketplaces.some((mp) => {
      if (mp.scope === "instance") return false;
      const orig = originalBySource.get(mp.source);
      return Boolean(mp.auto_load) !== Boolean(orig?.auto_load);
    });
  }, [allMarketplaces]);

  // Update lastKnownUpdatedAt when settings changes (org settings updated_at)
  useEffect(() => {
    // Use updated_at from orgAppSettings for 409 conflict handling
    if (orgAppSettings?.updated_at) {
      setLastKnownUpdatedAt(orgAppSettings.updated_at);
    }
  }, [orgAppSettings?.updated_at]);

  // Data loading effect - depends on settings and skills
  useEffect(() => {
    if (settings && skills) {
      // Backend returns scope directly on all marketplaces
      const all: MarketplaceRegistration[] = [
        ...(settings.inherited_marketplaces || []),
        ...(settings.registered_marketplaces || []),
      ];
      setAllMarketplaces(all);
      originalMarketplacesRef.current = all;

      // Build marketplace lookup for skills
      // All marketplaces have scope set from backend
      const marketplaceMap = new Map<
        string,
        { source: string; auto_load?: boolean; scope: string }
      >();
      for (const mp of settings.inherited_marketplaces || []) {
        marketplaceMap.set(mp.source, {
          source: mp.source,
          auto_load: mp.auto_load,
          scope: mp.scope || "instance",
        });
      }
      for (const mp of settings.registered_marketplaces || []) {
        marketplaceMap.set(mp.source, {
          source: mp.source,
          auto_load: mp.auto_load,
          scope: mp.scope || "personal",
        });
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
          // Check marketplaceMap for scope (already populated from inherited_marketplaces and registered_marketplaces)
          const marketplace =
            marketplaceMap.get(skill.name) || marketplaceMap.get(skill.source);
          if (marketplace) {
            repoUrl = marketplace.source;
            skillScope = marketplace.scope as "instance" | "org" | "personal";
          }
        }

        return {
          ...skill,
          id: skill.name,
          repository: repoUrl,
          scope: skillScope,
          isEnabled: !disabledSet.has(skill.name),
          isAutoLoad: !!marketplaceMap.get(skill.name)?.auto_load,
        };
      });

      // Fetch marketplace skills and merge with global/user skills
      const fetchMarketplaceSkills = async () => {
        // Use inherited_marketplaces for org/instance, registered for personal
        const allRegisteredMarketplaces: MarketplaceRegistration[] = [
          ...(settings.registered_marketplaces || []),
          ...(settings.inherited_marketplaces || []),
        ];

        if (allRegisteredMarketplaces.length === 0) {
          setSkillsState(mappedSkills);
          originalSkillsRef.current = mappedSkills;
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

                // Determine scope from marketplaceMap (already has scope info from backend)
                const marketplace = allRegisteredMarketplaces.find(
                  (mp) => skill.source === `marketplace:${mp.name}`,
                );
                const mpWithScope = marketplace
                  ? marketplaceMap.get(marketplace.source)
                  : undefined;

                // Scope should always come from backend; fallback is defensive only
                const skillScope =
                  (mpWithScope?.scope as "instance" | "org" | "personal") ||
                  "personal";

                marketplaceSkills.push({
                  ...skill,
                  id: skill.name,
                  repository: marketplace?.source || skill.source,
                  scope: skillScope,
                  isEnabled: !disabledSet.has(skill.name),
                  isAutoLoad: !!marketplace?.auto_load,
                });
              }
            }

            // Merge global/user skills with marketplace skills (both already deduplicated)
            const combinedSkills = [...mappedSkills, ...marketplaceSkills];
            setSkillsState(combinedSkills);
            originalSkillsRef.current = combinedSkills;
          } else {
            setSkillsState(mappedSkills);
            originalSkillsRef.current = mappedSkills;
          }
        } catch (error) {
          // Fall back to just global/user skills
          setSkillsState(mappedSkills);
          originalSkillsRef.current = mappedSkills;
        }
      };

      fetchMarketplaceSkills();
    }
  }, [settings, skills]);

  const filteredSkills = useMemo(
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

  const typeOptions = useMemo(() => {
    const types = new Set(skillsState.map((s) => s.type));
    return [
      { key: "all", label: t(I18nKey.SETTINGS$ALL_TYPES) },
      ...Array.from(types).map((type) => ({
        key: type.toLowerCase(),
        label: type.charAt(0).toUpperCase() + type.slice(1),
      })),
    ];
  }, [skillsState, t]);

  const repositoryOptions = useMemo(() => {
    const repos = new Set(skillsState.map((s) => s.repository));
    return [
      { key: "all", label: t(I18nKey.SETTINGS$ALL_REPOSITORIES) },
      ...Array.from(repos).map((repo) => ({
        key: repo,
        label: repo,
      })),
    ];
  }, [skillsState, t]);

  // Handlers with useCallback
  const handleToggleSkillEnabled = useCallback((skillId: string) => {
    setSkillsState((prev) =>
      prev.map((s) =>
        s.id === skillId ? { ...s, isEnabled: !s.isEnabled } : s,
      ),
    );
  }, []);

  const handleSaveSkillChanges = useCallback(() => {
    const disabledSkills = skillsState
      .filter((s) => !s.isEnabled)
      .map((s) => s.name);

    skillMutations.saveDisabledSkills.mutate(disabledSkills, {
      onSuccess: () => {
        originalSkillsRef.current = skillsState;
      },
    });
  }, [skillsState, skillMutations]);

  const handleToggleMarketplaceAutoLoad = useCallback((source: string) => {
    setAllMarketplaces((prev) =>
      prev.map((m) =>
        m.source === source ? { ...m, auto_load: !m.auto_load } : m,
      ),
    );
  }, []);

  const handleSaveMarketplaceChanges = useCallback(() => {
    const personal = allMarketplaces
      .filter((mp) => mp.scope === "personal")
      .map(({ name, source, ref, repo_path, auto_load, scope }) => ({
        name,
        source,
        ref,
        repo_path,
        auto_load,
        scope,
      }));

    const org = allMarketplaces
      .filter((mp) => mp.scope === "org")
      .map(({ name, source, ref, repo_path, auto_load, scope }) => ({
        name,
        source,
        ref,
        repo_path,
        auto_load,
        scope,
      }));

    if (personal.length > 0) {
      marketplaceMutations.savePersonal.mutate(personal, {
        onSuccess: () => {
          originalMarketplacesRef.current = allMarketplaces;
        },
      });
    }

    if (org.length > 0) {
      marketplaceMutations.saveOrg.mutate(
        { marketplaces: org, lastKnownUpdatedAt },
        {
          onSuccess: () => {
            originalMarketplacesRef.current = allMarketplaces;
          },
        },
      );
    }
  }, [allMarketplaces, lastKnownUpdatedAt, marketplaceMutations]);

  const openAddModal = useCallback(() => {
    setModalMode("add");
    setSelectedMarketplace(null);
    setIsModalOpen(true);
  }, []);

  const openEditModal = useCallback((marketplace: MarketplaceRegistration) => {
    setModalMode("edit");
    setSelectedMarketplace(marketplace);
    setIsModalOpen(true);
  }, []);

  const openDeleteModal = useCallback(
    (marketplace: MarketplaceRegistration) => {
      setMarketplaceToDelete(marketplace);
      setIsDeleteModalOpen(true);
    },
    [],
  );

  const handleSaveMarketplace = useCallback(
    (data: {
      name: string;
      source: string;
      ref?: string;
      repo_path?: string;
      auto_load?: boolean;
      scope?: "instance" | "org" | "personal";
    }) => {
      const newMarketplace: MarketplaceRegistration = {
        name: data.name,
        source: data.source,
        ref: data.ref,
        repo_path: data.repo_path,
        auto_load: data.auto_load,
        scope: data.scope || activeScope,
      };

      if (activeScope === "org") {
        const existing = allMarketplaces.filter((mp) => mp.scope === "org");
        const index = existing.findIndex((mp) => mp.source === data.source);
        const updated =
          index >= 0
            ? existing.map((mp, i) => (i === index ? newMarketplace : mp))
            : [...existing, newMarketplace];

        marketplaceMutations.saveOrg.mutate(
          { marketplaces: updated, lastKnownUpdatedAt },
          { onSuccess: () => setIsModalOpen(false) },
        );
      } else {
        const existing = allMarketplaces.filter(
          (mp) => mp.scope === "personal",
        );
        const index = existing.findIndex((mp) => mp.source === data.source);
        const updated =
          index >= 0
            ? existing.map((mp, i) => (i === index ? newMarketplace : mp))
            : [...existing, newMarketplace];

        marketplaceMutations.savePersonal.mutate(updated, {
          onSuccess: () => setIsModalOpen(false),
        });
      }
    },
    [activeScope, allMarketplaces, lastKnownUpdatedAt, marketplaceMutations],
  );

  const handleDeleteMarketplace = useCallback(() => {
    if (!marketplaceToDelete) return;

    if (marketplaceToDelete.scope === "org") {
      marketplaceMutations.deleteOrg.mutate(
        { marketplaceSource: marketplaceToDelete.source, lastKnownUpdatedAt },
        {
          onSuccess: () => {
            setIsDeleteModalOpen(false);
            setMarketplaceToDelete(null);
          },
        },
      );
    } else if (marketplaceToDelete.scope === "personal") {
      marketplaceMutations.deletePersonal.mutate(marketplaceToDelete.source, {
        onSuccess: () => {
          setIsDeleteModalOpen(false);
          setMarketplaceToDelete(null);
        },
      });
    }
  }, [marketplaceToDelete, lastKnownUpdatedAt, marketplaceMutations]);

  // Permission helpers
  const canEditMarketplace = useCallback(
    (mp: MarketplaceRegistration) => {
      if (mp.scope === "instance") return false;
      if (mp.scope === "org") return isAdminOrOwner;
      return true;
    },
    [isAdminOrOwner],
  );

  const getAutoLoadToggleTitle = useCallback(
    (scope: "instance" | "org" | "personal") => {
      if (scope === "instance")
        return t(I18nKey.SETTINGS$MARKETPLACE_INSTANCE_READONLY);
      if (scope === "org" && !isAdminOrOwner)
        return t(I18nKey.SETTINGS$MARKETPLACE_ORG_REQUIRES_ADMIN);
      return undefined;
    },
    [isAdminOrOwner, t],
  );

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
              onClick={() => openAddModal()}
            >
              {t(I18nKey.SETTINGS$MARKETPLACE_ADD)}
            </BrandButton>
          </div>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$CONNECT_MARKETPLACES_DESCRIPTION)}
          </Typography.Paragraph>
        </div>

        <MarketplaceTable
          marketplaces={allMarketplaces}
          onToggleAutoLoad={handleToggleMarketplaceAutoLoad}
          onEdit={openEditModal}
          onDelete={openDeleteModal}
          canEdit={canEditMarketplace}
          getAutoLoadTitle={getAutoLoadToggleTitle}
          isAdminOrOwner={isAdminOrOwner}
        />
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

        <SkillsTable
          skills={filteredSkills}
          onToggle={handleToggleSkillEnabled}
          typeOptions={typeOptions}
          repositoryOptions={repositoryOptions}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onTypeChange={setSelectedType}
          onRepositoryChange={setSelectedRepository}
        />
      </section>

      <div className="flex gap-6 p-6 justify-end border-t border-tertiary/50 mt-4">
        <BrandButton
          testId="skills-save-button"
          variant="primary"
          type="button"
          isDisabled={isSaving || (!hasSkillChanges && !hasMarketplaceChanges)}
          onClick={() => {
            if (hasSkillChanges) handleSaveSkillChanges();
            if (hasMarketplaceChanges) handleSaveMarketplaceChanges();
          }}
        >
          {isSaving
            ? t(I18nKey.SETTINGS$SAVING)
            : t(I18nKey.SETTINGS$SAVE_CHANGES)}
        </BrandButton>
      </div>

      {/* Marketplace Modal */}
      <MarketplaceModal
        isOpen={isModalOpen}
        mode={modalMode}
        marketplace={
          selectedMarketplace
            ? {
                name: selectedMarketplace.name,
                source: selectedMarketplace.source,
                ref: selectedMarketplace.ref,
                repo_path: selectedMarketplace.repo_path,
                auto_load: selectedMarketplace.auto_load,
                scope: selectedMarketplace.scope,
              }
            : null
        }
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveMarketplace}
        isSaving={
          marketplaceMutations.saveOrg.isPending ||
          marketplaceMutations.savePersonal.isPending
        }
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
