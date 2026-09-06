import React from "react";
import {
  GraphPage,
  GraphPageHeader,
} from "#/components/features/graph/graph-page";
import { kanbanPageScrollShellClassName } from "#/utils/kanban-page-layout-classes";

export default function GraphRoute() {
  return (
    <main data-testid="graph-page" className={kanbanPageScrollShellClassName}>
      <header className="mb-4">
        <GraphPageHeader />
      </header>
      <GraphPage />
    </main>
  );
}
