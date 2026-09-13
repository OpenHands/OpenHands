import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AutomationAgentProfileSelector } from "./agent-profile-selector";

vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => ({
    data: { profiles: [{ id: "review-profile", name: "Reviewer" }] },
    isLoading: false,
  }),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, args?: { id?: string }) =>
      args?.id ? `Unavailable profile (${args.id})` : key,
  }),
}));
vi.mock("#/components/features/settings/settings-dropdown-input", () => ({
  SettingsDropdownInput: ({
    items,
    selectedKey,
    onSelectionChange,
  }: {
    items: { key: string; label: string }[];
    selectedKey: string;
    onSelectionChange: (key: string) => void;
  }) => (
    <select
      aria-label="Agent profile"
      value={selectedKey}
      onChange={(event) => onSelectionChange(event.target.value)}
    >
      {items.map((item) => (
        <option key={item.key} value={item.key}>
          {item.label}
        </option>
      ))}
    </select>
  ),
}));

describe("automation agent profile selection", () => {
  it("saves the stable profile ID and clears explicitly to the deployment default", () => {
    const onChange = vi.fn();
    render(<AutomationAgentProfileSelector value={null} onChange={onChange} />);
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "review-profile" },
    });
    expect(onChange).toHaveBeenLastCalledWith("review-profile");
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "__default__" },
    });
    expect(onChange).toHaveBeenLastCalledWith(null);
  });
  it("retains a deleted profile instead of silently selecting another", () => {
    render(
      <AutomationAgentProfileSelector
        value="deleted-profile"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("combobox")).toHaveValue("deleted-profile");
    expect(
      screen.getByText("Unavailable profile (deleted-profile)"),
    ).toBeInTheDocument();
  });
});
