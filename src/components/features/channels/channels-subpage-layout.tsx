import { Inbox, Radio, ScrollText } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import {
  CHANNELS_MEETILY_PATH,
  CHANNELS_MESSAGES_PAGE_PATH,
  CHANNELS_PATH,
} from "#/api/channel-service/channel-constants";
import {
  ManifestSubpageLayout,
  type SubPageNavItem,
} from "#/components/features/manifest/manifest-subpage-layout";
import { I18nKey } from "#/i18n/declaration";

export function useChannelSubPageNav(): {
  heading: string;
  items: SubPageNavItem[];
} {
  const { t } = useTranslation("openhands");
  return {
    heading: t(I18nKey.CHANNELS$NAV),
    items: [
      {
        to: CHANNELS_PATH,
        label: t(I18nKey.CHANNELS$CONNECTIONS),
        Icon: Radio,
        testId: "channels-navigation-channels",
      },
      {
        to: CHANNELS_MESSAGES_PAGE_PATH,
        label: t(I18nKey.CHANNELS$MESSAGES),
        Icon: Inbox,
        testId: "channels-navigation-messages",
      },
      {
        to: CHANNELS_MEETILY_PATH,
        label: t(I18nKey.MEETILY$TITLE),
        Icon: ScrollText,
        testId: "channels-navigation-meetily",
      },
    ],
  };
}

export function ChannelsSubpageLayout({ children }: { children: ReactNode }) {
  const nav = useChannelSubPageNav();
  return (
    <ManifestSubpageLayout
      heading={nav.heading}
      navTestIdBase="channels-navbar"
      items={nav.items}
    >
      {children}
    </ManifestSubpageLayout>
  );
}
