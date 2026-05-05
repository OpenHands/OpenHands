import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { ProfilesBody } from "#/components/features/settings/profiles-body";

const profiles: LlmProfileSummary[] = [
  { name: "p1", model: "openai/gpt-4o", base_url: null, api_key_set: true },
  { name: "p2", model: "anthropic/claude", base_url: null, api_key_set: true },
];

describe("ProfilesBody", () => {
  it("shows a loading spinner while isLoading is true", () => {
    render(
      <ProfilesBody
        isLoading
        loadError={null}
        profiles={[]}
        active={null}
        onActivate={vi.fn()}
        onEdit={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        isActivating={false}
      />,
    );

    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
  });

  it("shows the load-error paragraph when loadError is set", () => {
    render(
      <ProfilesBody
        isLoading={false}
        loadError={new Error("boom")}
        profiles={[]}
        active={null}
        onActivate={vi.fn()}
        onEdit={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        isActivating={false}
      />,
    );

    expect(
      screen.getByText("SETTINGS$PROFILES_LOAD_ERROR"),
    ).toBeInTheDocument();
  });

  it("shows the empty-state paragraph when no profiles are passed", () => {
    render(
      <ProfilesBody
        isLoading={false}
        loadError={null}
        profiles={[]}
        active={null}
        onActivate={vi.fn()}
        onEdit={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        isActivating={false}
      />,
    );

    expect(screen.getByText("SETTINGS$PROFILES_EMPTY")).toBeInTheDocument();
  });

  it("renders one ProfileRow per profile and marks the active one", () => {
    render(
      <ProfilesBody
        isLoading={false}
        loadError={null}
        profiles={profiles}
        active="p1"
        onActivate={vi.fn()}
        onEdit={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
        isActivating={false}
      />,
    );

    const rows = screen.getAllByTestId("profile-row");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("p1");
    expect(rows[0]).toHaveTextContent("SETTINGS$PROFILE_ACTIVE_BADGE");
    expect(rows[1]).not.toHaveTextContent("SETTINGS$PROFILE_ACTIVE_BADGE");
  });
});
