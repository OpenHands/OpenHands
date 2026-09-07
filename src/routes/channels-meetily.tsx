import { MeetilyImport } from "#/components/features/channels/meetily";
import { ChannelsSubpageLayout } from "#/components/features/channels/channels-subpage-layout";

export default function ChannelsMeetilyRoute() {
  return (
    <ChannelsSubpageLayout>
      <MeetilyImport />
    </ChannelsSubpageLayout>
  );
}
