import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ProfileNameInput } from "#/components/features/settings/llm-profiles/profile-name-input";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        "SETTINGS$PROFILE_NAME_LABEL": "Profile Name",
        "SETTINGS$PROFILE_NAME_PLACEHOLDER": "Enter profile name",
        "SETTINGS$PROFILE_NAME_RULE":
          "1-64 chars, start with alphanumeric, then alphanumerics or . _ -",
        "COMMON$OPTIONAL": "Optional",
      };
      return translations[key] || key;
    },
  }),
}));

describe("ProfileNameInput", () => {
  it("renders with label and input", () => {
    render(<ProfileNameInput value="" onChange={() => {}} />);

    expect(screen.getByText("Profile Name")).toBeInTheDocument();
    expect(screen.getByRole("textbox")).toBeInTheDocument();
  });

  it("shows the rule text below the input", () => {
    render(
      <ProfileNameInput value="" onChange={() => {}} ruleTestId="rule-text" />,
    );

    const rule = screen.getByTestId("rule-text");
    expect(rule).toHaveTextContent(
      "1-64 chars, start with alphanumeric, then alphanumerics or . _ -",
    );
  });

  it("calls onChange when typing", async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ProfileNameInput
        testId="profile-name"
        value=""
        onChange={handleChange}
      />,
    );

    const input = screen.getByTestId("profile-name");
    await user.type(input, "gpt-4");

    // Each character triggers onChange
    expect(handleChange).toHaveBeenCalledTimes(5);
    // Each call receives just that character since the component is controlled
    // and the test doesn't update the value prop
    expect(handleChange).toHaveBeenNthCalledWith(1, "g");
    expect(handleChange).toHaveBeenNthCalledWith(2, "p");
    expect(handleChange).toHaveBeenNthCalledWith(3, "t");
    expect(handleChange).toHaveBeenNthCalledWith(4, "-");
    expect(handleChange).toHaveBeenNthCalledWith(5, "4");
  });

  it("displays the current value", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value="my-profile"
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveValue("my-profile");
  });

  it("marks the input as valid for valid names", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value="valid-name"
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "aria-invalid",
      "false",
    );
  });

  it("marks the input as invalid for names starting with a special character", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value=".invalid-start"
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
  });

  it("treats an empty optional value as valid", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value=""
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "aria-invalid",
      "false",
    );
  });

  it("marks whitespace-only values as invalid", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value="   "
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
  });

  it("marks names containing disallowed characters as invalid", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value="name with spaces"
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
  });

  it("can be disabled", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value=""
        onChange={() => {}}
        isDisabled
      />,
    );

    expect(screen.getByTestId("profile-name")).toBeDisabled();
  });

  it("shows custom placeholder", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value=""
        onChange={() => {}}
        placeholder="Custom placeholder"
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "placeholder",
      "Custom placeholder",
    );
  });

  it("shows default placeholder when not provided", () => {
    render(
      <ProfileNameInput
        testId="profile-name"
        value=""
        onChange={() => {}}
      />,
    );

    expect(screen.getByTestId("profile-name")).toHaveAttribute(
      "placeholder",
      "Enter profile name",
    );
  });

  it("shows optional label when isOptional is true", () => {
    render(
      <ProfileNameInput value="" onChange={() => {}} isOptional />,
    );

    expect(screen.getByText("Profile Name (Optional)")).toBeInTheDocument();
  });

  it("shows regular label when isOptional is false", () => {
    render(
      <ProfileNameInput value="" onChange={() => {}} isOptional={false} />,
    );

    expect(screen.getByText("Profile Name")).toBeInTheDocument();
    expect(screen.queryByText(/optional/i)).not.toBeInTheDocument();
  });

  describe("boundary conditions", () => {
    it("accepts exactly 64 character names", () => {
      const name64 = "a".repeat(64);
      render(
        <ProfileNameInput
          testId="profile-name"
          value={name64}
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "false",
      );
    });

    it("rejects 65 character names", () => {
      const name65 = "a".repeat(65);
      render(
        <ProfileNameInput
          testId="profile-name"
          value={name65}
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "true",
      );
    });

    it("accepts names starting with numbers", () => {
      render(
        <ProfileNameInput
          testId="profile-name"
          value="1profile"
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "false",
      );
    });

    it("accepts names with all allowed special characters", () => {
      render(
        <ProfileNameInput
          testId="profile-name"
          value="valid.name_with-chars"
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "false",
      );
    });

    it("rejects names starting with special characters", () => {
      render(
        <ProfileNameInput
          testId="profile-name"
          value=".invalid"
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "true",
      );
    });

    it("rejects names starting with hyphen", () => {
      render(
        <ProfileNameInput
          testId="profile-name"
          value="-invalid"
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "true",
      );
    });

    it("rejects names starting with underscore", () => {
      render(
        <ProfileNameInput
          testId="profile-name"
          value="_invalid"
          onChange={() => {}}
        />,
      );
      expect(screen.getByTestId("profile-name")).toHaveAttribute(
        "aria-invalid",
        "true",
      );
    });
  });
});
