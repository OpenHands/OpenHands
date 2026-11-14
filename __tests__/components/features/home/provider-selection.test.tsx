import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GitProviderDropdown } from "#/components/features/home/git-provider-dropdown/git-provider-dropdown";
import type { Provider } from "#/types/settings";
import type { GitProviderOption } from "#/utils/git-provider-option";
import { getProviderName } from "#/utils/utils";

const useTranslationMock = vi.hoisted(() => vi.fn());

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: unknown) => {
    useTranslationMock(namespace);
    return {
      t: (key: string) =>
        ({
          COMMON$SELECT_PROVIDER_PLACEHOLDER: "Select provider",
          COMMON$TOGGLE_MENU: "Toggle menu",
        })[key] ?? key,
    };
  },
}));

const ALL_PROVIDERS: Provider[] = [
  "github",
  "gitlab",
  "bitbucket",
  "bitbucket_data_center",
  "azure_devops",
  "forgejo",
];

const option = (provider: Provider): GitProviderOption => ({
  provider,
  gitProviderId: null,
  label: getProviderName(provider),
});

const toOptions = (providers: Provider[]) => providers.map(option);

type DropdownProps = React.ComponentProps<typeof GitProviderDropdown>;

function createProps(overrides: Partial<DropdownProps> = {}): DropdownProps {
  return {
    options: toOptions(ALL_PROVIDERS),
    onChange: vi.fn(),
    ...overrides,
  };
}

function ControlledDropdown({ onChange, value, ...props }: DropdownProps) {
  const [selectedOption, setSelectedOption] =
    React.useState<GitProviderOption | null>(value ?? null);

  return (
    <GitProviderDropdown
      {...props}
      value={selectedOption}
      onChange={(next) => {
        setSelectedOption(next);
        onChange?.(next);
      }}
    />
  );
}

describe("git provider selection", () => {
  it("includes the controlled provider in the initial markup", () => {
    const emptyMarkup = renderToStaticMarkup(
      <GitProviderDropdown options={toOptions(ALL_PROVIDERS)} />,
    );
    const selectedMarkup = renderToStaticMarkup(
      <GitProviderDropdown
        options={toOptions(ALL_PROVIDERS)}
        value={option("gitlab")}
      />,
    );

    expect(emptyMarkup).toContain('value=""');
    expect(selectedMarkup).toContain("pl-6");
    expect(selectedMarkup).toContain("min-w-3.5");
  });

  it("lists every provider with its user-facing name and selects one", async () => {
    const user = userEvent.setup();
    useTranslationMock.mockClear();
    const props = createProps({
      className: "provider-root",
      inputClassName: "provider-input",
      toggleButtonClassName: "provider-toggle-icon",
      itemClassName: "provider-option",
    });

    render(<ControlledDropdown {...props} />);

    const input = screen.getByTestId("git-provider-dropdown");
    expect(useTranslationMock).toHaveBeenNthCalledWith(1, "openhands");
    expect(input).toHaveAttribute("placeholder", "Select provider");
    expect(input).toHaveAttribute("readonly");
    expect(input.parentElement?.parentElement).toHaveClass(
      "relative",
      "provider-root",
    );
    expect(input).toHaveClass(
      "provider-input",
      "w-29.5",
      "border",
      "rounded",
      "text-inherit",
      "bg-tertiary",
      "placeholder:text-muted",
      "focus:outline-none",
      "focus:ring-0",
      "focus:border-border-input",
      "disabled:bg-tertiary",
      "disabled:cursor-not-allowed",
      "disabled:opacity-60",
      "pl-1.5",
      "pr-px",
      "cursor-pointer",
      "text-xs",
      "font-normal",
      "leading-5",
    );
    expect(screen.queryByTestId("dropdown-loading")).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Toggle menu" }).querySelector("svg"),
    ).toHaveClass("provider-toggle-icon");

    await user.click(input);

    expect(
      screen.getAllByRole("option").map((option) => option.textContent),
    ).toEqual([
      "GitHub",
      "GitLab",
      "Bitbucket",
      "Bitbucket Data Center",
      "Azure DevOps",
      "Forgejo",
    ]);
    expect(screen.getByRole("option", { name: "GitLab" })).toHaveClass(
      "provider-option",
    );

    await user.click(screen.getByRole("option", { name: "GitLab" }));

    expect(props.onChange).toHaveBeenCalledWith(option("gitlab"));
    expect(props.onChange).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(input).toHaveValue("GitLab"));
    expect(input).toHaveClass("pl-6");
    expect(input.previousElementSibling).toHaveClass(
      "absolute",
      "left-2",
      "z-10",
    );
    expect(input.previousElementSibling?.querySelector("svg")).toHaveClass(
      "min-w-3.5",
      "h-3.5",
    );
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("updates the available options when the providers prop changes", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <GitProviderDropdown
        {...createProps({ options: toOptions(["github"]) })}
      />,
    );
    const input = screen.getByTestId("git-provider-dropdown");

    await user.click(input);
    expect(screen.getAllByRole("option")).toHaveLength(1);
    expect(screen.getByRole("option", { name: "GitHub" })).toBeVisible();
    await user.keyboard("{Escape}");

    rerender(
      <GitProviderDropdown
        {...createProps({ options: toOptions(["gitlab", "azure_devops"]) })}
      />,
    );
    await user.click(input);

    expect(
      screen.getAllByRole("option").map((option) => option.textContent),
    ).toEqual(["GitLab", "Azure DevOps"]);
  });

  it("keeps provider options uniquely keyed when their order changes", async () => {
    const user = userEvent.setup();
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    const { rerender } = render(
      <GitProviderDropdown
        {...createProps({ options: toOptions(["github", "gitlab"]) })}
      />,
    );
    const input = screen.getByTestId("git-provider-dropdown");

    try {
      await user.click(input);
      rerender(
        <GitProviderDropdown
          {...createProps({ options: toOptions(["gitlab", "github"]) })}
        />,
      );

      expect(
        screen.getAllByRole("option").map((option) => option.textContent),
      ).toEqual(["GitLab", "GitHub"]);
      expect(consoleError).not.toHaveBeenCalledWith(
        expect.stringContaining('unique "key" prop'),
      );
    } finally {
      consoleError.mockRestore();
    }
  });

  it("allows a different provider to be selected without a callback", async () => {
    const user = userEvent.setup();
    render(<GitProviderDropdown options={toOptions(["github", "gitlab"])} />);
    const input = screen.getByTestId("git-provider-dropdown");

    await user.click(input);
    const runtimeErrors: unknown[] = [];
    const preventRuntimeError = (event: ErrorEvent) => {
      event.preventDefault();
      runtimeErrors.push(event.error);
    };
    window.addEventListener("error", preventRuntimeError);
    try {
      await user.click(screen.getByRole("option", { name: "GitLab" }));
    } finally {
      window.removeEventListener("error", preventRuntimeError);
    }

    expect(runtimeErrors).toEqual([]);
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
    await waitFor(() => expect(input).toHaveValue(""));
  });

  it("reopens a selected provider with the complete provider list", async () => {
    const user = userEvent.setup();
    render(
      <GitProviderDropdown {...createProps({ value: option("gitlab") })} />,
    );
    const input = screen.getByTestId("git-provider-dropdown");
    await waitFor(() => expect(input).toHaveValue("GitLab"));

    await user.click(input);

    expect(screen.getAllByRole("option")).toHaveLength(ALL_PROVIDERS.length);
    expect(screen.getByRole("option", { name: "GitHub" })).toBeVisible();
    expect(screen.getByRole("option", { name: "Forgejo" })).toBeVisible();
  });

  it("tracks controlled provider changes and clearing", async () => {
    const props = createProps({ value: option("github"), isLoading: true });
    const { rerender } = render(<GitProviderDropdown {...props} />);
    const input = screen.getByTestId("git-provider-dropdown");

    await waitFor(() => expect(input).toHaveValue("GitHub"));
    expect(screen.getByTestId("dropdown-loading").parentElement).toHaveClass(
      "right-11",
    );

    rerender(<GitProviderDropdown {...props} value={option("azure_devops")} />);
    await waitFor(() => expect(input).toHaveValue("Azure DevOps"));

    rerender(<GitProviderDropdown {...props} value={null} />);
    await waitFor(() => expect(input).toHaveValue(""));
    expect(input).not.toHaveClass("pl-6");
    expect(screen.getByTestId("dropdown-loading").parentElement).toHaveClass(
      "right-6",
    );
  });

  it("prevents interaction while disabled", async () => {
    const user = userEvent.setup();
    render(<GitProviderDropdown {...createProps({ disabled: true })} />);

    const input = screen.getByTestId("git-provider-dropdown");
    const toggle = screen.getByRole("button", { name: "Toggle menu" });
    expect(input).toBeDisabled();
    expect(toggle).toBeDisabled();

    await user.click(input);
    await user.click(toggle);
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("shows loading and error feedback with a custom placeholder", () => {
    render(
      <GitProviderDropdown
        {...createProps({
          placeholder: "Choose a source",
          isLoading: true,
          errorMessage: "Providers could not be loaded",
        })}
      />,
    );

    expect(screen.getByTestId("git-provider-dropdown")).toHaveAttribute(
      "placeholder",
      "Choose a source",
    );
    expect(screen.getByTestId("dropdown-loading")).toBeVisible();
    expect(screen.getByTestId("dropdown-error")).toHaveTextContent(
      "Providers could not be loaded",
    );
  });

  it("shows an empty state when no providers are available", async () => {
    const user = userEvent.setup();
    render(
      <GitProviderDropdown {...createProps({ options: toOptions([]) })} />,
    );

    await user.click(screen.getByTestId("git-provider-dropdown"));

    expect(screen.getByTestId("git-provider-dropdown-empty")).toHaveTextContent(
      "No providers available",
    );
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("keeps its read-only list intact when the user types", async () => {
    const user = userEvent.setup();
    render(<GitProviderDropdown {...createProps()} />);
    const input = screen.getByTestId("git-provider-dropdown");

    await user.click(input);
    await user.keyboard("git");

    expect(input).toHaveValue("");
    expect(screen.getAllByRole("option")).toHaveLength(ALL_PROVIDERS.length);
  });

  it("dismisses the menu with Escape or an outside click", async () => {
    const user = userEvent.setup();
    render(
      <>
        <button type="button">Outside</button>
        <GitProviderDropdown {...createProps()} />
      </>,
    );
    const input = screen.getByTestId("git-provider-dropdown");

    await user.click(input);
    expect(screen.getAllByRole("option")).toHaveLength(ALL_PROVIDERS.length);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("option")).not.toBeInTheDocument();

    await user.click(input);
    expect(screen.getAllByRole("option")).toHaveLength(ALL_PROVIDERS.length);
    await user.click(screen.getByRole("button", { name: "Outside" }));
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("clears the selected provider when Escape is pressed while closed", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <ControlledDropdown
        {...createProps({ value: option("github"), onChange })}
      />,
    );
    const input = screen.getByTestId("git-provider-dropdown");
    await waitFor(() => expect(input).toHaveValue("GitHub"));

    await user.click(input);
    await user.keyboard("{Escape}");
    expect(input).toHaveValue("GitHub");

    await user.keyboard("{Escape}");
    await waitFor(() => expect(input).toHaveValue(""));
    expect(onChange).toHaveBeenCalledWith(null);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("allows the current provider to be reselected without a callback", async () => {
    const user = userEvent.setup();
    render(
      <GitProviderDropdown
        options={toOptions(["forgejo"])}
        value={option("forgejo")}
      />,
    );
    const input = screen.getByTestId("git-provider-dropdown");
    await waitFor(() => expect(input).toHaveValue("Forgejo"));

    await user.click(input);
    await user.click(screen.getByRole("option", { name: "Forgejo" }));

    await waitFor(() => expect(input).toHaveValue("Forgejo"));
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });

  it("marks the current provider as selected when the menu opens", async () => {
    const user = userEvent.setup();
    render(
      <GitProviderDropdown {...createProps({ value: option("bitbucket") })} />,
    );
    const input = screen.getByTestId("git-provider-dropdown");
    await waitFor(() => expect(input).toHaveValue("Bitbucket"));

    await user.click(input);

    const listbox = screen.getByRole("listbox");
    const selectedOption = within(listbox).getByRole("option", {
      name: "Bitbucket",
    });
    const unselectedOption = within(listbox).getByRole("option", {
      name: "GitHub",
    });
    expect(selectedOption).toHaveAttribute("aria-selected", "true");
    expect(selectedOption).toHaveClass("bg-interactive-selected");
    expect(selectedOption).not.toHaveClass("hover:bg-interactive-hover");
    expect(unselectedOption).toHaveAttribute("aria-selected", "false");
    expect(unselectedOption).toHaveClass("hover:bg-interactive-hover");
    expect(unselectedOption).not.toHaveClass("bg-interactive-selected");
  });

  describe("connections of the same provider type", () => {
    const workGitLab: GitProviderOption = {
      provider: "gitlab",
      gitProviderId: 4,
      label: "Work GitLab",
    };
    const selfHostedGitLab: GitProviderOption = {
      provider: "gitlab",
      gitProviderId: 9,
      label: "Self-hosted GitLab",
    };

    it("lists each connection by its title and reports the one picked", async () => {
      const user = userEvent.setup();
      const onChange = vi.fn();
      render(
        <ControlledDropdown
          {...createProps({
            options: [workGitLab, selfHostedGitLab],
            onChange,
          })}
        />,
      );

      await user.click(screen.getByTestId("git-provider-dropdown"));
      expect(
        screen.getAllByRole("option").map((item) => item.textContent),
      ).toEqual(["Work GitLab", "Self-hosted GitLab"]);

      await user.click(
        screen.getByRole("option", { name: "Self-hosted GitLab" }),
      );

      expect(onChange).toHaveBeenCalledExactlyOnceWith(selfHostedGitLab);
      await waitFor(() =>
        expect(screen.getByTestId("git-provider-dropdown")).toHaveValue(
          "Self-hosted GitLab",
        ),
      );
    });

    it("marks only the matching connection as selected", async () => {
      const user = userEvent.setup();
      render(
        <GitProviderDropdown
          options={[workGitLab, selfHostedGitLab]}
          value={{ ...selfHostedGitLab }}
        />,
      );

      await user.click(screen.getByTestId("git-provider-dropdown"));

      expect(
        screen.getByRole("option", { name: "Self-hosted GitLab" }),
      ).toHaveAttribute("aria-selected", "true");
      expect(
        screen.getByRole("option", { name: "Work GitLab" }),
      ).toHaveAttribute("aria-selected", "false");
    });
  });
});
