import type { ComponentType, SVGProps } from "react";
import {
  Bot,
  Briefcase,
  Building2,
  CircleUserRound,
  CloudCog,
  Flag,
  Folder,
  FolderGit2,
  FolderKey,
  GitBranch,
  GitPullRequest,
  Globe2,
  Hash,
  House,
  IdCard,
  KeyRound,
  Layers,
  Link2,
  Mails,
  MessagesSquare,
  Plug,
  SquareKanban,
  Tag,
  Ticket,
  UsersRound,
  Waypoints,
  Webhook,
  Wrench,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { FaBitbucket, FaGithub, FaGitlab } from "react-icons/fa6";
import type { IconType } from "react-icons/lib";
import SlackIcon from "#/icons/slack.svg?react";

/**
 * Any icon renderable inside a tag chip / overflow row. Lucide, react-icons,
 * and local SVG React components all work as long as they accept ``className``
 * and inherit ``currentColor`` for the muted chip text.
 */
export type ConversationTagIcon =
  | LucideIcon
  | IconType
  | ComponentType<SVGProps<SVGSVGElement>>;

/**
 * Tag keys whose value names a git host / chat source — resolve the icon from
 * the value (e.g. ``git_provider: github`` → GitHub mark) instead of the key.
 */
const SOURCE_VALUE_DRIVEN_TAG_KEYS = new Set([
  "origin",
  "source",
  "git_provider",
]);

/**
 * App-mode keys resolve from the mode value (``work`` → briefcase) when known,
 * otherwise fall back to {@link Layers}.
 */
const APP_MODE_TAG_KEYS = new Set(["appmode", "app_mode", "mode"]);

/**
 * Icons for well-known conversation tag keys. Unknown keys fall back to
 * {@link Tag}. Value-driven keys additionally resolve via the value maps below.
 */
const KEY_ICONS: Record<string, ConversationTagIcon> = {
  origin: Waypoints,
  source: Link2,
  git_provider: FaGithub,
  owner: CircleUserRound,
  user: CircleUserRound,
  author: CircleUserRound,
  assignee: CircleUserRound,
  env: CloudCog,
  environment: CloudCog,
  repo: FolderGit2,
  repository: FolderGit2,
  repo_name: FolderGit2,
  branch: GitBranch,
  selected_branch: GitBranch,
  archiveworkspacepath: Folder,
  workspace: Folder,
  working_dir: Folder,
  team: UsersRound,
  org: Building2,
  organization: Building2,
  channel: Hash,
  email: Mails,
  automation: Zap,
  webhook: Webhook,
  agent: Bot,
  project: SquareKanban,
  ticket: Ticket,
  issue: Ticket,
  pr: GitPullRequest,
  pull_request: GitPullRequest,
  priority: Flag,
  status: IdCard,
  id: KeyRound,
  integration: Plug,
  appmode: Layers,
  app_mode: Layers,
  mode: Layers,
  worktools: Wrench,
  work_tools: Wrench,
  tools: Wrench,
  workwsid: FolderKey,
  work_wsid: FolderKey,
  wsid: FolderKey,
  workspace_id: FolderKey,
};

/**
 * Value-specific icons for source / provider stamps (``origin``,
 * ``git_provider``, …), keyed by the lowercase stamp value.
 */
const SOURCE_VALUE_ICONS: Record<string, ConversationTagIcon> = {
  slack: SlackIcon,
  discord: MessagesSquare,
  github: FaGithub,
  gitlab: FaGitlab,
  bitbucket: FaBitbucket,
  bitbucket_data_center: FaBitbucket,
  azure_devops: GitBranch,
  email: Mails,
  mail: Mails,
  api: Plug,
  webhook: Webhook,
  automation: Zap,
  review: GitPullRequest,
  linear: SquareKanban,
  web: Globe2,
  ui: Globe2,
  canvas: Globe2,
};

/**
 * Value-specific icons for app-mode stamps (``Appmode: work`` → briefcase).
 */
const APP_MODE_VALUE_ICONS: Record<string, ConversationTagIcon> = {
  work: Briefcase,
  personal: House,
  home: House,
  private: House,
};

/**
 * Pick an icon that matches a conversation tag. Prefer value-specific icons
 * for source/provider and app-mode keys; otherwise map by key; finally fall
 * back to ``Tag``.
 */
export function getConversationTagIcon(
  key: string,
  value: string,
): ConversationTagIcon {
  const normalizedKey = key.trim().toLowerCase();
  const normalizedValue = value.trim().toLowerCase();

  if (SOURCE_VALUE_DRIVEN_TAG_KEYS.has(normalizedKey)) {
    return (
      SOURCE_VALUE_ICONS[normalizedValue] ?? KEY_ICONS[normalizedKey] ?? Tag
    );
  }

  if (APP_MODE_TAG_KEYS.has(normalizedKey)) {
    return (
      APP_MODE_VALUE_ICONS[normalizedValue] ??
      KEY_ICONS[normalizedKey] ??
      Layers
    );
  }

  return KEY_ICONS[normalizedKey] ?? Tag;
}
