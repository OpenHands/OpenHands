import { render, screen } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryRouter } from "react-router";
import { TaskSuggestions } from "#/components/features/home/tasks/task-suggestions";

// Mock translation
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

// Mock suggested tasks hook
vi.mock("#/hooks/query/use-suggested-tasks", () => ({
  useSuggestedTasks: () => ({
    data: [],
    isLoading: false,
  }),
}));

// Mock config hook
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: vi.fn(),
}));

// Mock user providers hook
vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: vi.fn(),
}));

import { useConfig } from "#/hooks/query/use-config";
import { useUserProviders } from "#/hooks/use-user-providers";

describe("TaskSuggestions empty states", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("OSS mode + no providers → shows Git provider empty state", () => {
    (useConfig as any).mockReturnValue({
      data: { app_mode: "oss" },
    });

    (useUserProviders as any).mockReturnValue({
      providers: [],
    });

    render(
      <MemoryRouter>
        <TaskSuggestions />
      </MemoryRouter>
    );

    expect(screen.getByText("TASKS$NO_GIT_PROVIDERS_TITLE")).toBeInTheDocument();
    expect(screen.getByText("TASKS$NO_GIT_PROVIDERS_DESCRIPTION")).toBeInTheDocument();
    expect(screen.getByText("TASKS$NO_GIT_PROVIDERS_CTA")).toBeInTheDocument();
  });

  it("OSS mode + providers exist but no tasks → shows no tasks message", () => {
    (useConfig as any).mockReturnValue({
      data: { app_mode: "oss" },
    });

    (useUserProviders as any).mockReturnValue({
      providers: [{ id: "github" }],
    });

    render(
      <MemoryRouter>
        <TaskSuggestions />
      </MemoryRouter>
    );

    expect(screen.getByText("TASKS$NO_TASKS_AVAILABLE")).toBeInTheDocument();
  });
});
