import { describe, expect, it } from "vitest";
import {
  CircleUserRound,
  CloudCog,
  Folder,
  FolderGit2,
  GitBranch,
  Tag,
  Waypoints,
} from "lucide-react";
import { FaGithub } from "react-icons/fa6";
import SlackIcon from "#/icons/slack.svg?react";
import { getConversationTagIcon } from "#/components/features/conversation-panel/conversation-card/conversation-tag-icons";

describe("getConversationTagIcon", () => {
  it("maps known keys to related icons", () => {
    expect(getConversationTagIcon("owner", "alice")).toBe(CircleUserRound);
    expect(getConversationTagIcon("env", "prod")).toBe(CloudCog);
    expect(getConversationTagIcon("repo_name", "org/repo")).toBe(FolderGit2);
    expect(getConversationTagIcon("selected_branch", "main")).toBe(GitBranch);
    expect(
      getConversationTagIcon("archiveworkspacepath", "/workspace/project"),
    ).toBe(Folder);
  });

  it("prefers source brand icons for origin and git_provider values", () => {
    expect(getConversationTagIcon("origin", "slack")).toBe(SlackIcon);
    expect(getConversationTagIcon("git_provider", "GitHub")).toBe(FaGithub);
  });

  it("falls back to the key icon for unknown sources", () => {
    expect(getConversationTagIcon("origin", "custom-bot")).toBe(Waypoints);
  });

  it("falls back to Tag for unrecognized keys", () => {
    expect(getConversationTagIcon("mystery", "value")).toBe(Tag);
  });
});
