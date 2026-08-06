import { describe, expect, it } from "vitest";
import {
  Briefcase,
  CircleUserRound,
  CloudCog,
  Folder,
  FolderGit2,
  FolderKey,
  GitBranch,
  House,
  Layers,
  Tag,
  Waypoints,
  Wrench,
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
    expect(getConversationTagIcon("worktools", "browser")).toBe(Wrench);
    expect(
      getConversationTagIcon("Workwsid", "cad8995a71e54b12b156d1d153be062f"),
    ).toBe(FolderKey);
  });

  it("uses mode-specific icons for Appmode values", () => {
    expect(getConversationTagIcon("Appmode", "work")).toBe(Briefcase);
    expect(getConversationTagIcon("app_mode", "personal")).toBe(House);
    expect(getConversationTagIcon("mode", "custom")).toBe(Layers);
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
