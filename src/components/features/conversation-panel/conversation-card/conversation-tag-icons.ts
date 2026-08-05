import type { ComponentType, SVGProps } from "react";
import {
  Bot,
  Building2,
  CircleUserRound,
  CloudCog,
  Flag,
  Folder,
  FolderGit2,
  GitBranch,
  GitPullRequest,
  Globe2,
  Hash,
  IdCard,
  KeyRound,
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
const VALUE_DRIVEN_TAG_KEYS = new Set(["origin", "source", "git_provider"]);

/**
 * Icons for well-known conversation tag keys. Unknown keys fall back to
 * {@link Tag}. Value-driven keys (see {@link VALUE_DRIVEN_TAG_KEYS}) additionally
 * resolve via {@link SOURCE_VALUE_ICONS}.
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
 * Pick an icon that matches a conversation tag. Prefer value-specific icons
 * for source/provider keys; otherwise map by key; finally fall back to ``Tag``.
 */
export function getConversationTagIcon(
  key: string,
  value: string,
): ConversationTagIcon {
  const normalizedKey = key.trim().toLowerCase();
  const normalizedValue = value.trim().toLowerCase();

  if (VALUE_DRIVEN_TAG_KEYS.has(normalizedKey)) {
    return (
      SOURCE_VALUE_ICONS[normalizedValue] ?? KEY_ICONS[normalizedKey] ?? Tag
    );
  }

  return KEY_ICONS[normalizedKey] ?? Tag;
}
