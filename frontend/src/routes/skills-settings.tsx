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
    // Name is the marketplace identity (unique across scopes).
    const originalByName = new Map(original.map((mp) => [mp.name, mp]));
    return allMarketplaces.some((mp) => {
      if (mp.scope === "instance") return false;
      const orig = originalByName.get(mp.name);
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
    if (!settings || !skills) return undefined;
    // Guards against out-of-order async results when settings/skills change
    // (or the component unmounts) before the marketplace preview resolves.
    let cancelled = false;

    // Backend returns scope directly on all marketplaces
    const all: MarketplaceRegistration[] = [
      ...(settings.inherited_marketplaces || []),
      ...(settings.registered_marketplaces || []),
    ];
    setAllMarketplaces(all);
    originalMarketplacesRef.current = all;

    // Marketplace lookup for skills (keyed by source; scope comes from backend).
    const marketplaceMap = new Map<
      string,
      { source: string; auto_load?: boolean; scope: string }
    >();
    for (const mp of all) {
      marketplaceMap.set(mp.source, {
        source: mp.source,
        auto_load: mp.auto_load,
        scope: mp.scope ?? "personal",
      });
    }

    // Map global/user/repo skills with marketplace info
    const disabledSet = new Set(settings.disabled_skills || []);
    const mappedSkills: SkillWithState[] = skills.map((skill) => {
      let repoUrl = skill.source;
      let skillScope: "instance" | "org" | "personal" = "personal";

      if (skill.source === "global") {
        skillScope = "instance";
      } else if (skill.source === "user") {
        skillScope = "personal";
      } else {
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
      if (all.length === 0) {
        if (!cancelled) {
          setSkillsState(mappedSkills);
          originalSkillsRef.current = mappedSkills;
        }
        return;
      }

      try {
        const preview = await SkillsService.getMarketplaceSkills(all);
        if (cancelled) return;

        if (preview.errors && preview.errors.length > 0) {
          preview.errors.forEach((error) => {
            displayErrorToast(`Marketplace error: ${error}`);
          });
        }

        // Seed with global/user/repo names so a marketplace skill sharing a name
        // with an existing skill never creates a duplicate row / React key.
        const seenSkillNames = new Set<string>(mappedSkills.map((s) => s.name));
        const marketplaceSkills: SkillWithState[] = [];

        for (const skill of preview.skills) {
          if (!seenSkillNames.has(skill.name)) {
            seenSkillNames.add(skill.name);

            const marketplace = all.find(
              (mp) => skill.source === `marketplace:${mp.name}`,
            );
            const mpWithScope = marketplace
              ? marketplaceMap.get(marketplace.source)
              : undefined;
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

        // Plugins advertised by marketplace manifests, shown at the plugin
        // level (bundled skills are not expanded). Enablement/auto-load is
        // governed by the parent marketplace, so these rows are read-only here.
        const marketplacePlugins: SkillWithState[] = (
          preview.plugins ?? []
        ).map((plugin) => {
          const mp = marketplaceMap.get(plugin.source);
          const pluginScope =
            (mp?.scope as "instance" | "org" | "personal") || "personal";
          const autoLoad = !!mp?.auto_load;
          return {
            name: plugin.name,
            type: "plugin",
            source: `marketplace:${plugin.marketplace}`,
            id: `plugin:${plugin.marketplace}:${plugin.name}`,
            repository: plugin.source,
            scope: pluginScope,
            isEnabled: autoLoad,
            isAutoLoad: autoLoad,
          };
        });

        const combinedSkills = [
          ...mappedSkills,
          ...marketplaceSkills,
          ...marketplacePlugins,
        ];
        setSkillsState(combinedSkills);
        originalSkillsRef.current = combinedSkills;
      } catch {
        if (!cancelled) {
          setSkillsState(mappedSkills);
          originalSkillsRef.current = mappedSkills;
        }
      }
    };

    fetchMarketplaceSkills();
    return () => {
      cancelled = true;
    };
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
    // Plugin rows are read-only previews whose toggle mirrors the marketplace's
    // auto-load; they must never leak into the user's disabled_skills list.
    const disabledSkills = skillsState
      .filter((s) => s.type !== "plugin" && !s.isEnabled)
      .map((s) => s.name);

    skillMutations.saveDisabledSkills.mutate(disabledSkills, {
      onSuccess: () => {
        originalSkillsRef.current = skillsState;
      },
    });
  }, [skillsState, skillMutations]);

  const handleToggleMarketplaceAutoLoad = useCallback((name: string) => {
    setAllMarketplaces((prev) =>
      prev.map((m) =>
        m.name === name ? { ...m, auto_load: !m.auto_load } : m,
      ),
    );
  }, []);

  const handleSaveMarketplaceChanges = useCallback(() => {
    const original = originalMarketplacesRef.current;
    const origByName = new Map(original.map((mp) => [mp.name, mp]));
    const autoLoadChanged = (mp: MarketplaceRegistration) =>
      Boolean(mp.auto_load) !== Boolean(origByName.get(mp.name)?.auto_load);

    // Never send the backend-derived `scope`.
    const toPayload = (mp: MarketplaceRegistration) => ({
      name: mp.name,
      source: mp.source,
      ref: mp.ref,
      repo_path: mp.repo_path,
      auto_load: mp.auto_load,
    });

    // Only persist a scope whose toggles actually changed, so a personal edit
    // never re-writes the org list (churning org.updated_at / causing conflicts).
    const personalChanged = allMarketplaces.some(
      (mp) => mp.scope === "personal" && autoLoadChanged(mp),
    );
    const orgChanged = allMarketplaces.some(
      (mp) => mp.scope === "org" && autoLoadChanged(mp),
    );

    if (personalChanged) {
      const personal = allMarketplaces
        .filter((mp) => mp.scope === "personal")
        .map(toPayload);
      marketplaceMutations.savePersonal.mutate(personal, {
        onSuccess: () => {
          originalMarketplacesRef.current = allMarketplaces;
        },
      });
    }

    // Org marketplaces are admin/owner-only (also enforced server-side).
    if (orgChanged && isAdminOrOwner) {
      const org = allMarketplaces
        .filter((mp) => mp.scope === "org")
        .map(toPayload);
      marketplaceMutations.saveOrg.mutate(
        { marketplaces: org, lastKnownUpdatedAt },
        {
          onSuccess: () => {
            originalMarketplacesRef.current = allMarketplaces;
          },
        },
      );
    }
  }, [
    allMarketplaces,
    isAdminOrOwner,
    lastKnownUpdatedAt,
    marketplaceMutations,
  ]);

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
      scope?: "org" | "personal";
    }) => {
      // Edit keeps the item's existing scope; add uses the chosen scope
      // (admins in an org) or defaults to personal.
      const targetScope: "org" | "personal" =
        modalMode === "edit" && selectedMarketplace
          ? (selectedMarketplace.scope as "org" | "personal")
          : data.scope || "personal";

      const entry: MarketplaceRegistration = {
        name: data.name,
        source: data.source,
        ref: data.ref,
        repo_path: data.repo_path,
        auto_load: data.auto_load,
        scope: targetScope,
      };

      // Never send the backend-derived `scope`.
      const toPayload = (mp: MarketplaceRegistration) => ({
        name: mp.name,
        source: mp.source,
        ref: mp.ref,
        repo_path: mp.repo_path,
        auto_load: mp.auto_load,
      });

      const scopeList = allMarketplaces.filter(
        (mp) => mp.scope === targetScope,
      );
      // On edit, replace the row identified by its original name (name is the
      // identity and may itself be edited); on add, append.
      const updatedList =
        modalMode === "edit" && selectedMarketplace
          ? scopeList.map((mp) =>
              mp.name === selectedMarketplace.name ? entry : mp,
            )
          : [...scopeList, entry];
      const updated = updatedList.map(toPayload);

      if (targetScope === "org") {
        marketplaceMutations.saveOrg.mutate(
          { marketplaces: updated, lastKnownUpdatedAt },
          { onSuccess: () => setIsModalOpen(false) },
        );
      } else {
        marketplaceMutations.savePersonal.mutate(updated, {
          onSuccess: () => setIsModalOpen(false),
        });
      }
    },
    [
      modalMode,
      selectedMarketplace,
      allMarketplaces,
      lastKnownUpdatedAt,
      marketplaceMutations,
    ],
  );

  const handleDeleteMarketplace = useCallback(() => {
    if (!marketplaceToDelete) return;

    if (marketplaceToDelete.scope === "org") {
      marketplaceMutations.deleteOrg.mutate(
        { marketplaceName: marketplaceToDelete.name, lastKnownUpdatedAt },
        {
          onSuccess: () => {
            setIsDeleteModalOpen(false);
            setMarketplaceToDelete(null);
          },
        },
      );
    } else if (marketplaceToDelete.scope === "personal") {
      marketplaceMutations.deletePersonal.mutate(marketplaceToDelete.name, {
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
      <div className="flex h-64 items-center justify-center">
        <div className="animate-pulse text-sm text-tertiary-alt">
          Loading...
        </div>
      </div>
    );
  }

  const hasUnsavedChanges = hasSkillChanges || hasMarketplaceChanges;

  return (
    <div className="flex min-h-full flex-col">
      <div className="flex flex-col gap-10 pb-6">
        {/* Marketplaces */}
        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="flex flex-col gap-1">
              <Typography.H2>{t(I18nKey.SETTINGS$MARKETPLACES)}</Typography.H2>
              <Typography.Paragraph className="max-w-2xl text-sm text-tertiary-alt">
                {t(I18nKey.SETTINGS$CONNECT_MARKETPLACES_DESCRIPTION)}
              </Typography.Paragraph>
            </div>
            <BrandButton
              testId="add-marketplace-button"
              variant="primary"
              type="button"
              onClick={() => openAddModal()}
            >
              {t(I18nKey.SETTINGS$MARKETPLACE_ADD)}
            </BrandButton>
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

        {/* Available Skills */}
        <section className="flex flex-col gap-4">
          <div className="flex flex-col gap-1">
            <Typography.H2>
              {t(I18nKey.SETTINGS$SKILLS_AND_PLUGINS)}
            </Typography.H2>
            <Typography.Paragraph className="max-w-2xl text-sm text-tertiary-alt">
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
      </div>

      {/* Sticky action bar — keeps Save reachable without scrolling to the end */}
      <div className="sticky bottom-0 z-10 mt-auto flex items-center justify-end gap-4 border-t border-tertiary bg-base/95 py-4 backdrop-blur-sm">
        {hasUnsavedChanges && (
          <span className="text-sm text-tertiary-alt">
            {t(I18nKey.SETTINGS$UNSAVED_CHANGES)}
          </span>
        )}
        <BrandButton
          testId="skills-save-button"
          variant="primary"
          type="button"
          isDisabled={isSaving || !hasUnsavedChanges}
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
        allowScopeSelection={isAdminOrOwner && activeScope === "org"}
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
