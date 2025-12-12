import { Tab, Tabs } from "@heroui/react";
import { useTranslation } from "react-i18next";
import { MicroagentManagementRepositories } from "./microagent-management-repositories";
import { I18nKey } from "#/i18n/declaration";
import { useMicroagentManagementStore } from "#/state/microagent-management-store";

interface MicroagentManagementSidebarTabsProps {
  isSearchLoading?: boolean;
}

export function MicroagentManagementSidebarTabs({
  isSearchLoading = false,
}: MicroagentManagementSidebarTabsProps) {
  const { t } = useTranslation();

  const { repositories, personalRepositories, organizationRepositories } =
    useMicroagentManagementStore();

  return (
    <div className="flex w-full flex-col">
      <Tabs
        aria-label="Options"
        classNames={{
          base: "py-6",
          tabList:
            "w-full bg-transparent border border-[#ffffff40] rounded-[6px]",
          tab: "px-2 h-[22px]",
          tabContent: "text-white text-[12px] font-normal",
          panel: "p-0",
          cursor: "bg-[#C9B97480] rounded-sm",
        }}
      >
        <Tab key="personal" title={t(I18nKey.COMMON$PERSONAL)}>
          <MicroagentManagementRepositories
            repositories={personalRepositories}
            tabType="personal"
            isSearchLoading={isSearchLoading}
          />
        </Tab>
        <Tab key="repositories" title={t(I18nKey.COMMON$REPOSITORIES)}>
          <MicroagentManagementRepositories
            repositories={repositories}
            tabType="repositories"
            isSearchLoading={isSearchLoading}
          />
        </Tab>
        <Tab key="organizations" title={t(I18nKey.COMMON$ORGANIZATIONS)}>
          <MicroagentManagementRepositories
            repositories={organizationRepositories}
            tabType="organizations"
            isSearchLoading={isSearchLoading}
          />
        </Tab>
      </Tabs>
    </div>
  );
}
