import React from "react";
import { ChannelsOverview } from "#/components/features/channels/channels-overview";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";

export default function ChannelsRoute() {
  return (
    <main
      data-testid="channels-page"
      className={kanbanPageScrollShellClassName}
    >
      <ChannelsOverview />
    </main>
  );
}
