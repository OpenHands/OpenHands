import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Suggestions } from "#/components/features/suggestions/suggestions";

describe("Suggestions", () => {
  const firstSuggestion = {
    label: "first-suggestion",
    value: "value-of-first-suggestion",
  };
  const secondSuggestion = {
    label: "second-suggestion",
    value: "value-of-second-suggestion",
  };
  const suggestions = [firstSuggestion, secondSuggestion];

  const onSuggestionClickMock = vi.fn();

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render suggestions", () => {
    render(
      <Suggestions
        suggestions={suggestions}
        onSuggestionClick={onSuggestionClickMock}
      />,
    );

    expect(screen.getByTestId("suggestions")).toBeInTheDocument();
    const suggestionElements = screen.getAllByTestId("suggestion");

    expect(suggestionElements).toHaveLength(2);
    expect(suggestionElements[0]).toHaveTextContent("first-suggestion");
    expect(suggestionElements[1]).toHaveTextContent("second-suggestion");
  });

  it("should call onSuggestionClick when clicking a suggestion", async () => {
    const user = userEvent.setup();
    render(
      <Suggestions
        suggestions={suggestions}
        onSuggestionClick={onSuggestionClickMock}
      />,
    );

    const suggestionElements = screen.getAllByTestId("suggestion");

    await user.click(suggestionElements[0]);
    expect(onSuggestionClickMock).toHaveBeenCalledWith(
      "value-of-first-suggestion",
    );

    await user.click(suggestionElements[1]);
    expect(onSuggestionClickMock).toHaveBeenCalledWith(
      "value-of-second-suggestion",
    );
  });
});
