import React from "react";
import {
  RoutingPage,
  RoutingPageHeader,
} from "#/components/features/routing/routing-page";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";

export default function RoutingRoute() {
  return (
    <main data-testid="routing-page" className={kanbanPageScrollShellClassName}>
      <header className="mb-4">
        <RoutingPageHeader />
      </header>
      <RoutingPage />
    </main>
  );
}
