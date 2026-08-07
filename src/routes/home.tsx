import { PrefetchPageLinks, useLocation } from "react-router";
import { HomeChatLauncher } from "#/components/features/home/home-chat-launcher";
import { FeaturedAutomationsDemo } from "#/components/features/home/featured-automations-demo";
import { LlmNotConfiguredBanner } from "#/components/features/home/llm-not-configured-banner";
import {
  isOnboardingPreviewActive,
  OnboardingHost,
} from "#/components/features/onboarding";

<PrefetchPageLinks page="/conversations/:conversationId" />;

function HomeScreen() {
  const location = useLocation();
  const isPreview = isOnboardingPreviewActive(location.search);

  return (
    <div
      data-testid="home-screen"
      className="custom-scrollbar-always h-full overflow-y-auto rounded-xl bg-transparent px-4 md:px-0 lg:px-[42px]"
    >
      {import.meta.env.VITE_FEATURED_AUTOMATIONS_DEMO === "true" ? (
        <div className="pb-3 pt-4">
          <FeaturedAutomationsDemo />
        </div>
      ) : null}

      <div className="md:px-4 lg:px-0">
        <LlmNotConfiguredBanner />
      </div>

      <HomeChatLauncher />

      {!isPreview ? <OnboardingHost /> : null}
    </div>
  );
}

export default HomeScreen;
