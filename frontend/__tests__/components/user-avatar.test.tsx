import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { UserAvatar } from "#/components/features/sidebar/user-avatar";

describe("UserAvatar", () => {
  it("(default) should render the placeholder avatar when the user is logged out", () => {
    render(<UserAvatar />);
    expect(screen.getByTestId("user-avatar")).toBeInTheDocument();
    expect(
      screen.getByLabelText("USER$AVATAR_PLACEHOLDER"),
    ).toBeInTheDocument();
  });

  it("should display the user's avatar when available", () => {
    render(<UserAvatar avatarUrl="https://example.com/avatar.png" />);

    expect(screen.getByAltText("AVATAR$ALT_TEXT")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("USER$AVATAR_PLACEHOLDER"),
    ).not.toBeInTheDocument();
  });

  it("should display a loading spinner instead of an avatar when isLoading is true", () => {
    const { rerender } = render(<UserAvatar />);
    expect(screen.queryByTestId("loading-spinner")).not.toBeInTheDocument();
    expect(
      screen.getByLabelText("USER$AVATAR_PLACEHOLDER"),
    ).toBeInTheDocument();

    rerender(<UserAvatar isLoading />);
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    expect(
      screen.queryByLabelText("USER$AVATAR_PLACEHOLDER"),
    ).not.toBeInTheDocument();

    rerender(
      <UserAvatar avatarUrl="https://example.com/avatar.png" isLoading />,
    );
    expect(screen.getByTestId("loading-spinner")).toBeInTheDocument();
    expect(screen.queryByAltText("AVATAR$ALT_TEXT")).not.toBeInTheDocument();
  });
});
