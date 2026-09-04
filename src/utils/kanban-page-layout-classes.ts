import { cn } from "#/utils/utils";

/**
 * Kanban and project-init are top-level outlet siblings of the sidebar, not
 * settings-like pages with a 260px aside. Match the home-screen gutter
 * (`px-4` / `lg:px-[42px]`) rather than `settingsLikeMainScrollClassName`.
 */
export const kanbanPageGutterClassName = "px-4 pt-6 pb-4 lg:px-[42px]";

export const kanbanPageShellClassName = cn(
  "flex h-full min-h-0 flex-col overflow-hidden",
  kanbanPageGutterClassName,
);

export const kanbanPageScrollShellClassName = cn(
  "flex h-full min-h-0 flex-col overflow-y-auto custom-scrollbar-always",
  kanbanPageGutterClassName,
);
