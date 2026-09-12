import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import SkillsService from "#/api/skills-service";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useSkills } from "./use-skills";

const SKILLS_KEY_PREFIX = ["skills", null] as const;

function renderUseSkills() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
  return { queryClient, ...renderHook(() => useSkills(), { wrapper }) };
}

describe("useSkills", () => {
  it("scopes its cache entry to the active backend", async () => {
    vi.spyOn(SkillsService, "getSkills").mockResolvedValue([]);

    const { queryClient, result } = renderUseSkills();
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const [key] = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
    // Unscoped, a second backend reads this same entry and shows the first
    // backend's skills until the 10-minute staleTime expires.
    expect(key.slice(0, SKILLS_KEY_PREFIX.length)).toEqual([
      ...SKILLS_KEY_PREFIX,
    ]);
    expect(key.length).toBe(SKILLS_KEY_PREFIX.length + 2);
  });
});
