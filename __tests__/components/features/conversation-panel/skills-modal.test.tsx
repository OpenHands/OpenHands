import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";

const skillsQuery = vi.hoisted(() => ({
  data: [
    { name: "repo-skill", type: "agentskills", source: "project" },
    { name: "personal-skill", type: "knowledge", source: "user" },
    { name: "public-skill", type: "knowledge", source: "public" },
  ],
  isLoading: false,
  isError: false,
  isRefetching: false,
  refetch: vi.fn(),
}));

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => skillsQuery,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("SkillsModal", () => {
  it("retains available-skill provenance on an active conversation", () => {
    render(<SkillsModal onClose={vi.fn()} />);

    expect(screen.getByTestId("skills-modal")).toBeInTheDocument();
    expect(screen.getByText("SKILLS_MODAL$SECTION_PROJECT")).toBeVisible();
    expect(screen.getByText("SKILLS_MODAL$SECTION_USER")).toBeVisible();
    expect(screen.getByText("SKILLS_MODAL$SECTION_PUBLIC")).toBeVisible();
    expect(screen.getByText("repo-skill")).toBeVisible();
    expect(screen.getByText("personal-skill")).toBeVisible();
    expect(screen.getByText("public-skill")).toBeVisible();
  });
});
