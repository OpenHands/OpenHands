import { GuideMessage } from "./guide-message";
import { HomeHeaderTitle } from "./home-header-title";

export function HomeHeader() {
  return (
    <header className="flex flex-col items-center gap-12">
      <GuideMessage />
      <HomeHeaderTitle />
    </header>
  );
}
