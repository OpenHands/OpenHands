import React from "react";
import { PrefetchPageLinks } from "react-router";
import { HomeHeader } from "#/components/features/home/home-header/home-header";
import { RepoConnector } from "#/components/features/home/repo-connector";
import { TaskSuggestions } from "#/components/features/home/tasks/task-suggestions";
import { GitRepository } from "#/types/git";
import { NewConversation } from "#/components/features/home/new-conversation/new-conversation";
import { RecentConversations } from "#/components/features/home/recent-conversations/recent-conversations";
import { HomepageCTA } from "#/components/features/home/homepage-cta";
import { isCTADismissed } from "#/utils/local-storage";
import { useAppMode } from "#/hooks/use-app-mode";

<PrefetchPageLinks page="/conversations/:conversationId" />;

function HomeScreen() {
  const { isEnterpriseCloud } = useAppMode();
  const [selectedRepo, setSelectedRepo] = React.useState<GitRepository | null>(
    null,
  );

  const [shouldShowCTA, setShouldShowCTA] = React.useState(
    () => !isCTADismissed("homepage"),
  );

  return (
    <div
      data-testid="home-screen"
      className="px-0 pt-4 bg-transparent h-full flex flex-col pt-[35px] overflow-y-auto rounded-xl lg:px-[42px] lg:pt-[42px] custom-scrollbar-always"
    >
      <HomeHeader />

      <div className="pt-[25px] flex justify-center">
        <div
          className="flex flex-col gap-5 px-6 sm:max-w-full sm:min-w-full md:flex-row lg:px-0 lg:max-w-[703px] lg:min-w-[703px]"
          data-testid="home-screen-new-conversation-section"
        >
          <RepoConnector onRepoSelection={(repo) => setSelectedRepo(repo)} />
          <NewConversation />
        </div>
      </div>

      <div className="pt-4 flex sm:justify-start md:justify-center">
        <div
          className="flex flex-col gap-5 px-6 md:flex-row min-w-full md:max-w-full lg:px-0 lg:max-w-[703px] lg:min-w-[703px]"
          data-testid="home-screen-recent-conversations-section"
        >
          <RecentConversations />
          <TaskSuggestions filterFor={selectedRepo} />
        </div>
      </div>

      {isEnterpriseCloud && shouldShowCTA && (
        <div className="fixed bottom-4 right-8 z-50 md:bottom-6 md:right-12">
          <HomepageCTA setShouldShowCTA={setShouldShowCTA} />
        </div>
      )}
    </div>
  );
}

export default HomeScreen;
