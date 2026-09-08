import type { ComponentType, SVGProps } from "react";
import {
  CloudCog,
  FileText,
  PenLine,
  Rss,
  StickyNote,
  Webhook,
  Zap,
} from "lucide-react";
import CalendarIcon from "#/icons/calendar.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import ListIcon from "#/icons/list.svg?react";
import PuzzleIcon from "#/icons/puzzle.svg?react";
import TargetIcon from "#/icons/target.svg?react";
import TerminalIcon from "#/icons/terminal.svg?react";
import type { SetupFieldType } from "#/manifests/types";
import type { PreviewFieldKind } from "./automation-preview-order";

type IconComponent = ComponentType<SVGProps<SVGSVGElement>>;

/**
 * One concept keeps one icon across the app. These follow the choices already
 * made in `detail/configuration-section.tsx` (repositories, schedule, event
 * source, plugins) and in `conversation-card/conversation-tag-icons.ts`
 * (environment, webhook) — change them together or not at all.
 */
const SETUP_TYPE_ICONS: Record<SetupFieldType, IconComponent> = {
  cron: CalendarIcon,
  timezone: GlobeIcon,
  textarea: ListIcon,
  "repo-picker": GitBranchIcon,
  text: DocumentIcon,
  select: ListIcon,
};

const SETUP_NAME_ICONS: Record<string, IconComponent> = {
  name: PenLine,
  title: PenLine,
  feeds: Rss,
  topics: TargetIcon,
  summary: FileText,
  notes: StickyNote,
  webhook: Webhook,
  webhooks: Webhook,
  environment: CloudCog,
};

function isSetupFieldType(kind: PreviewFieldKind): kind is SetupFieldType {
  return kind in SETUP_TYPE_ICONS;
}

/** Falls back to the generic field icon for kinds a manifest cannot declare. */
export function setupPreviewFieldIcon(
  name: string,
  kind: PreviewFieldKind,
): IconComponent {
  return (
    SETUP_NAME_ICONS[name] ??
    (isSetupFieldType(kind) ? SETUP_TYPE_ICONS[kind] : DocumentIcon)
  );
}

export const ImportNameIcon = PenLine;
export const ImportPromptIcon = TerminalIcon;
export const ImportPluginsIcon = PuzzleIcon;
export const ImportScheduleIcon = CalendarIcon;
export const ImportEventIcon = Zap;
