import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { useActiveAcpProfileDetail } from "#/hooks/query/use-active-acp-profile-detail";

const listProfilesMock = vi.fn();
const getProfileMock = vi.fn();

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    listProfiles: (...args: unknown[]) => listProfilesMock(...args),
    getProfile: (...args: unknown[]) => getProfileMock(...args),
  },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));

function Probe() {
  const profile = useActiveAcpProfileDetail();
  return (
    <div data-testid="acp-detail">
      {profile ? `${profile.acp_server}:${profile.acp_model}` : "none"}
    </div>
  );
}

describe("useActiveAcpProfileDetail", () => {
  beforeEach(() => {
    listProfilesMock.mockReset();
    getProfileMock.mockReset();
  });

  it("resolves the active ACP profile's server and model on the home page", async () => {
    listProfilesMock.mockResolvedValue({
      profiles: [{ id: "id-claude", name: "claude", agent_kind: "acp" }],
      active_agent_profile_id: "id-claude",
    });
    getProfileMock.mockResolvedValue({
      profile: {
        id: "id-claude",
        name: "claude",
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_model: "sonnet",
      },
    });

    renderWithProviders(<Probe />, { navigation: { conversationId: null } });

    expect(await screen.findByText("claude-code:sonnet")).toBeInTheDocument();
    expect(getProfileMock).toHaveBeenCalledWith("claude");
  });

  it("does not fetch the detail while a conversation is open", async () => {
    listProfilesMock.mockResolvedValue({
      profiles: [{ id: "id-claude", name: "claude", agent_kind: "acp" }],
      active_agent_profile_id: "id-claude",
    });

    renderWithProviders(<Probe />, {
      navigation: { conversationId: "conv-1" },
    });

    await waitFor(() => expect(listProfilesMock).toHaveBeenCalled());
    expect(getProfileMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("acp-detail")).toHaveTextContent("none");
  });

  it("does not fetch the detail when the active profile is OpenHands", async () => {
    listProfilesMock.mockResolvedValue({
      profiles: [{ id: "id-oh", name: "default", agent_kind: "openhands" }],
      active_agent_profile_id: "id-oh",
    });

    renderWithProviders(<Probe />, { navigation: { conversationId: null } });

    await waitFor(() => expect(listProfilesMock).toHaveBeenCalled());
    expect(getProfileMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("acp-detail")).toHaveTextContent("none");
  });
});
