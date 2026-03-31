import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import {
  ClaimButton,
  getButtonState,
} from "#/components/features/org/claim-button";
import type { GitOrg } from "#/hooks/organizations/use-git-conversation-routing";

const createOrg = (overrides: Partial<GitOrg> = {}): GitOrg => ({
  id: "1",
  provider: "GitHub",
  name: "TestOrg",
  status: "unclaimed",
  ...overrides,
});

describe("getButtonState", () => {
  it("returns 'claiming' during claiming transition regardless of hover", () => {
    expect(getButtonState("claiming", false)).toBe("claiming");
    expect(getButtonState("claiming", true)).toBe("claiming");
  });

  it("returns 'disconnecting' during disconnecting transition regardless of hover", () => {
    expect(getButtonState("disconnecting", false)).toBe("disconnecting");
    expect(getButtonState("disconnecting", true)).toBe("disconnecting");
  });

  it("returns 'disconnect' when claimed and hovered", () => {
    expect(getButtonState("claimed", true)).toBe("disconnect");
  });

  it("returns 'claimed' when claimed and not hovered", () => {
    expect(getButtonState("claimed", false)).toBe("claimed");
  });

  it("returns 'unclaimed' when unclaimed", () => {
    expect(getButtonState("unclaimed", false)).toBe("unclaimed");
    expect(getButtonState("unclaimed", true)).toBe("unclaimed");
  });
});

describe("ClaimButton", () => {
  it("calls onClaim when clicking an unclaimed org", async () => {
    // Arrange
    const onClaim = vi.fn();
    const org = createOrg({ status: "unclaimed" });
    renderWithProviders(
      <ClaimButton org={org} onClaim={onClaim} onDisconnect={vi.fn()} />,
    );
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("claim-button-1"));

    // Assert
    expect(onClaim).toHaveBeenCalledWith("1");
  });

  it("calls onDisconnect when clicking a claimed org", async () => {
    // Arrange
    const onDisconnect = vi.fn();
    const org = createOrg({ status: "claimed" });
    renderWithProviders(
      <ClaimButton org={org} onClaim={vi.fn()} onDisconnect={onDisconnect} />,
    );
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("claim-button-1"));

    // Assert
    expect(onDisconnect).toHaveBeenCalledWith("1");
  });

  it("does not call handlers when button is disabled during claiming", async () => {
    // Arrange
    const onClaim = vi.fn();
    const onDisconnect = vi.fn();
    const org = createOrg({ status: "claiming" });
    renderWithProviders(
      <ClaimButton org={org} onClaim={onClaim} onDisconnect={onDisconnect} />,
    );
    const user = userEvent.setup();

    // Act
    await user.click(screen.getByTestId("claim-button-1"));

    // Assert
    expect(onClaim).not.toHaveBeenCalled();
    expect(onDisconnect).not.toHaveBeenCalled();
    expect(screen.getByTestId("claim-button-1")).toBeDisabled();
  });

  it("shows 'Disconnect' label on hover when claimed", async () => {
    // Arrange
    const org = createOrg({ status: "claimed" });
    renderWithProviders(
      <ClaimButton org={org} onClaim={vi.fn()} onDisconnect={vi.fn()} />,
    );
    const user = userEvent.setup();

    // Act
    await user.hover(screen.getByTestId("claim-button-1"));

    // Assert
    expect(screen.getByTestId("claim-button-1")).toHaveTextContent(
      "ORG$DISCONNECT",
    );
  });
});
