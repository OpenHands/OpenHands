import { PrefetchPageLinks } from "react-router";
import { HomeChatLauncher } from "#/components/features/home/home-chat-launcher";
import { LlmNotConfiguredBanner } from "#/components/features/home/llm-not-configured-banner";

<PrefetchPageLinks page="/conversations/:conversationId" />;

function HomeScreen() {
  return (
    <div
      data-testid="home-screen"
      className="custom-scrollbar-always h-full overflow-y-auto rounded-xl bg-transparent px-4 md:px-0 lg:px-[42px]"
    >
      <div className="md:px-4 lg:px-0">
        <LlmNotConfiguredBanner />
      </div>

      <HomeChatLauncher />
    </div>
  );
}

export default HomeScreen;
