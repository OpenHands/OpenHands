import React from "react";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { SettingsDropdownInput } from "#/components/features/settings/settings-dropdown-input";
import { cn } from "#/utils/utils";
import { useSaveOrgAppSettings } from "#/hooks/mutation/use-save-org-app-settings";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useSettings } from "#/hooks/query/use-settings";
import { useSkills } from "#/hooks/query/use-skills";
import { MarketplaceRegistration, SkillWithState } from "#/types/settings";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import { I18nKey } from "#/i18n/declaration";
import { organizationService } from "#/api/organization-service/organization-service.api";

// Validation patterns for marketplace sources (must match backend patterns)
const MARKETPLACE_PATTERNS = {
  github: /^github:[a-zA-Z0-9_.-]+[/][a-zA-Z0-9_.-]+$/,
  gitUrl:
    /^(https?:\/\/|git@|ssh:\/\/|git:\/\/)[a-zA-Z0-9_.-]+[:/][a-zA-Z0-9_./-]+$/,
  localPath: /^[a-zA-Z0-9_][a-zA-Z0-9_./-]*$/,
};

/**
 * Validates if a source string is a valid marketplace source format.
 * Must match validation patterns in openhands/storage/data_models/settings.py
 */
function isValidMarketplaceSource(source: string): boolean {
  const trimmed = source.trim();
  if (!trimmed) return false;

  return (
    MARKETPLACE_PATTERNS.github.test(trimmed) ||
    MARKETPLACE_PATTERNS.gitUrl.test(trimmed) ||
    MARKETPLACE_PATTERNS.localPath.test(trimmed)
  );
}

function WhiteToggle({
  isToggled,
  onClick,
  "aria-label": ariaLabel,
}: {
  isToggled: boolean;
  onClick?: () => void;
  "aria-label"?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="cursor-pointer"
      aria-label={ariaLabel}
    >
      <div
        className={cn(
          "w-12 h-6 rounded-xl flex items-center p-1.5",
          isToggled && "justify-end bg-white",
          !isToggled && "justify-start bg-base-secondary",
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

function SkillsSettingsScreen() {
  const { t } = useTranslation();
  const { isPending: isSaving } = useSaveSettings();
  const { mutate: saveOrgAppSettings } = useSaveOrgAppSettings();
  const { data: settings, isLoading: settingsLoading } = useSettings();
  const { data: skills, isLoading: skillsLoading } = useSkills();

  // Fetch org app settings with updated_at for optimistic locking
  const { data: orgAppSettings } = useQuery({
    queryKey: ["organization-app-settings"],
    queryFn: () => organizationService.getOrganizationAppSettings(),
    retry: false,
  });

  const [skillsState, setSkillsState] = React.useState<SkillWithState[]>([]);
  const [hasChanges, setHasChanges] = React.useState(false);

  const [repositories, setRepositories] = React.useState<
    MarketplaceRegistration[]
  >([]);
  const [repositoryUrl, setRepositoryUrl] = React.useState("");

  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedType, setSelectedType] = React.useState<string | null>(null);
  const [selectedRepository, setSelectedRepository] = React.useState<
    string | null
  >(null);

  // Track expected_updated_at for optimistic locking
  const [expectedUpdatedAt, setExpectedUpdatedAt] = React.useState<
    string | null
  >(null);

  React.useEffect(() => {
    if (orgAppSettings?.updated_at) {
      setExpectedUpdatedAt(orgAppSettings.updated_at);
    }
  }, [orgAppSettings?.updated_at]);

  React.useEffect(() => {
    if (settings && skills) {
      const disabledSet = new Set(settings.disabled_skills || []);
      const marketplaceMap = new Map(
        (settings.registered_marketplaces || []).map((mp) => [mp.source, mp]),
      );

      const mappedSkills: SkillWithState[] = skills.map((skill) => {
        let repoUrl = skill.source;
        if (skill.source !== "global" && skill.source !== "user") {
          const marketplace =
            marketplaceMap.get(skill.name) || marketplaceMap.get(skill.source);
          if (marketplace) {
            repoUrl = marketplace.source;
          }
        }

        return {
          ...skill,
          id: skill.name,
          repository: repoUrl,
          isEnabled: !disabledSet.has(skill.name),
          isAutoLoad: marketplaceMap.get(skill.name)?.auto_load === "all",
        };
      });

      setSkillsState(mappedSkills);
      setRepositories(settings.registered_marketplaces || []);
    }
  }, [settings, skills]);

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

  const handleToggleEnabled = (skillId: string) => {
    setSkillsState((prev) =>
      prev.map((skill) =>
        skill.id === skillId
          ? { ...skill, isEnabled: !skill.isEnabled }
          : skill,
      ),
    );
    setHasChanges(true);
  };

  const handleToggleAutoLoad = (skillId: string) => {
    setSkillsState((prev) =>
      prev.map((skill) =>
        skill.id === skillId
          ? { ...skill, isAutoLoad: !skill.isAutoLoad }
          : skill,
      ),
    );
    setHasChanges(true);
  };

  const handleSave = () => {
    // Build marketplace list from manually added repositories
    // and skills that have auto_load enabled
    const marketplaceMap = new Map<string, MarketplaceRegistration>();

    // Add manually added repositories (highest priority)
    for (const repo of repositories) {
      marketplaceMap.set(repo.source, repo);
    }

    // Add auto-load skills as marketplaces (if not already present)
    for (const skill of skillsState) {
      if (skill.isAutoLoad && !marketplaceMap.has(skill.repository)) {
        marketplaceMap.set(skill.repository, {
          source: skill.repository,
          name: skill.repository.split("/").pop() || skill.repository,
          auto_load: "all",
        });
      }
    }

    saveOrgAppSettings(
      {
        registered_marketplaces: Array.from(marketplaceMap.values()),
        expected_updated_at: expectedUpdatedAt,
      },
      {
        onSuccess: (data) => {
          // Update expected_updated_at for next save
          setExpectedUpdatedAt(data.updated_at);
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED));
          setHasChanges(false);
        },
        onError: (error) => {
          const errorMessage = retrieveAxiosErrorMessage(error);
          displayErrorToast(errorMessage || t(I18nKey.ERROR$GENERIC));
        },
      },
    );
  };

  const handleAddRepository = () => {
    const trimmedUrl = repositoryUrl.trim();
    if (!trimmedUrl) return;

    // Validate the source format before adding
    if (!isValidMarketplaceSource(trimmedUrl)) {
      displayErrorToast(
        "Invalid repository format. Use github:owner/repo, a git URL, or a relative path.",
      );
      return;
    }

    const newMarketplace: MarketplaceRegistration = {
      source: trimmedUrl,
      name: trimmedUrl.split("/").pop() || trimmedUrl,
      auto_load: "all",
    };
    setRepositories((prev) => [...prev, newMarketplace]);
    setRepositoryUrl("");
    setHasChanges(true);
  };

  const isLoading = settingsLoading || skillsLoading || !settings;

  const getSourceLabel = (source: string) => {
    if (source === "global") {
      return t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_INSTANCE);
    }
    if (source === "user") {
      return t(I18nKey.SETTINGS$MARKETPLACE_SCOPE_PERSONAL);
    }
    return source;
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <div className="mb-8">
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$ORG_SKILLS_TITLE)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$ORG_SKILLS_DESCRIPTION)}
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
          {t(I18nKey.SETTINGS$ORG_SKILLS_TITLE)}
        </Typography.H2>
        <Typography.Paragraph className="text-sm text-[#8c8c8c]">
          {t(I18nKey.SETTINGS$ORG_SKILLS_DESCRIPTION)}
        </Typography.Paragraph>
      </div>

      <section className="mb-8 flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$CONNECT_REPOSITORIES)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$CONNECT_REPOSITORIES_DESCRIPTION)}
          </Typography.Paragraph>
        </div>

        <div className="flex items-center gap-4">
          <input
            data-testid="repository-url-input"
            type="text"
            placeholder={t(I18nKey.SETTINGS$MARKETPLACE_SOURCE_PLACEHOLDER)}
            value={repositoryUrl}
            onChange={(e) => setRepositoryUrl(e.target.value)}
            className="bg-tertiary border border-[#717888] h-10 w-full rounded-sm p-2 placeholder:italic placeholder:text-tertiary-alt"
          />
          <button
            type="button"
            onClick={handleAddRepository}
            className="bg-white text-[#0D0F11] px-4 py-2 rounded-sm font-medium hover:opacity-80 cursor-pointer whitespace-nowrap"
          >
            {t(I18nKey.SETTINGS$MARKETPLACE_ADD)}
          </button>
        </div>

        <div className="border border-tertiary rounded-md overflow-hidden">
          <table className="w-full">
            <thead className="bg-base-secondary">
              <tr className="grid grid-cols-1 gap-4 items-start">
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$MARKETPLACE_SOURCE)}
                </th>
              </tr>
            </thead>
            <tbody>
              {repositories.map((repo) => (
                <tr
                  key={repo.source}
                  className="grid grid-cols-1 gap-4 items-start border-t border-tertiary"
                >
                  <td className="p-3 text-sm text-content-2 truncate min-w-0 text-tertiary-alt">
                    {repo.source}
                  </td>
                </tr>
              ))}
              {repositories.length === 0 && (
                <tr className="border-t border-tertiary">
                  <td
                    colSpan={1}
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

      <section className="flex flex-col gap-4">
        <div className="flex flex-col gap-1">
          <Typography.H2 className="mb-2">
            {t(I18nKey.SETTINGS$SKILLS_PERMISSIONS)}
          </Typography.H2>
          <Typography.Paragraph className="text-sm text-[#8c8c8c]">
            {t(I18nKey.SETTINGS$SKILLS_PERMISSIONS_DESCRIPTION)}
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
              <tr className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr] gap-4 items-start">
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
                  {t(I18nKey.SETTINGS$SOURCE)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$ENABLED)}
                </th>
                <th className="text-left p-3 text-sm font-medium uppercase text-tertiary-alt">
                  {t(I18nKey.SETTINGS$AUTO_LOAD)}
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredSkills.map((skill) => (
                <tr
                  key={skill.id}
                  className="grid grid-cols-[1fr_1fr_1fr_1fr_1fr_1fr] gap-4 items-center border-t border-tertiary"
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
                  <td className="p-3 text-sm text-tertiary-alt capitalize">
                    {getSourceLabel(skill.source)}
                  </td>
                  <td className="p-3">
                    <WhiteToggle
                      isToggled={skill.isEnabled}
                      onClick={() => handleToggleEnabled(skill.id)}
                      aria-label={`Toggle enabled for ${skill.name}`}
                    />
                  </td>
                  <td className="p-3">
                    <WhiteToggle
                      isToggled={skill.isAutoLoad}
                      onClick={() => handleToggleAutoLoad(skill.id)}
                      aria-label={`Toggle auto-load for ${skill.name}`}
                    />
                  </td>
                </tr>
              ))}
              {filteredSkills.length === 0 && (
                <tr className="border-t border-tertiary">
                  <td
                    colSpan={6}
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

      {hasChanges && (
        <div className="flex gap-6 p-6 justify-end border-t border-tertiary/50 mt-4">
          <BrandButton
            testId="skills-save-button"
            variant="primary"
            type="button"
            isDisabled={isSaving}
            onClick={handleSave}
          >
            {!isSaving && t(I18nKey.SETTINGS$SAVE_CHANGES)}
            {isSaving && t(I18nKey.SETTINGS$SAVING)}
          </BrandButton>
        </div>
      )}
    </div>
  );
}

export default SkillsSettingsScreen;
