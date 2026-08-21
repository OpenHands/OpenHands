import type { ComponentType, SVGProps } from "react";
import {
  FileText,
  GitBranch,
  Layers,
  PenLine,
  Rss,
  StickyNote,
  Webhook,
  Zap,
} from "lucide-react";
import CalendarIcon from "#/icons/calendar.svg?react";
import DocumentIcon from "#/icons/document.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import ListIcon from "#/icons/list.svg?react";
import PuzzleIcon from "#/icons/puzzle.svg?react";
import TargetIcon from "#/icons/target.svg?react";
import TerminalIcon from "#/icons/terminal.svg?react";
import type { SetupFieldType } from "#/manifests/types";

type IconComponent = ComponentType<SVGProps<SVGSVGElement>>;

const SETUP_TYPE_ICONS: Record<SetupFieldType, IconComponent> = {
  cron: CalendarIcon,
  timezone: GlobeIcon,
  textarea: ListIcon,
  "repo-picker": GitBranch,
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
  environment: Layers,
};

export function setupPreviewFieldIcon(
  name: string,
  type: SetupFieldType,
): IconComponent {
  return SETUP_NAME_ICONS[name] ?? SETUP_TYPE_ICONS[type];
}

export const ImportNameIcon = PenLine;
export const ImportPromptIcon = TerminalIcon;
export const ImportPluginsIcon = PuzzleIcon;
export const ImportScheduleIcon = CalendarIcon;
export const ImportEventIcon = Zap;
