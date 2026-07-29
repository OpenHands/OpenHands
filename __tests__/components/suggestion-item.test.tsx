import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SuggestionItem } from "#/components/features/suggestions/suggestion-item";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        SUGGESTIONS$TODO_APP: "ToDoリストアプリを開発する",
        LANDING$BUILD_APP_BUTTON: "プルリクエストを表示するアプリを開発する",
        SUGGESTIONS$HACKER_NEWS:
          "Hacker Newsのトップ記事を表示するbashスクリプトを作成する",
      };
      return translations[key] || key;
    },
  }),
}));

describe("SuggestionItem", () => {
  const suggestionItem = { label: "suggestion1", value: "a long text value" };
  const onClick = vi.fn();

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render a suggestion", () => {
    render(<SuggestionItem suggestion={suggestionItem} onClick={onClick} />);

    expect(screen.getByTestId("suggestion")).toBeInTheDocument();
    expect(screen.getByText(/suggestion1/i)).toBeInTheDocument();
  });

  it("should render a translated suggestion when using I18nKey", async () => {
    const translatedSuggestion = {
      label: I18nKey.SUGGESTIONS$TODO_APP,
      value: "todo app value",
    };

    render(
      <SuggestionItem suggestion={translatedSuggestion} onClick={onClick} />,
    );

    expect(screen.getByText("ToDoリストアプリを開発する")).toBeInTheDocument();
  });

  it("should call onClick when clicking a suggestion", async () => {
    const user = userEvent.setup();
    render(<SuggestionItem suggestion={suggestionItem} onClick={onClick} />);

    const suggestion = screen.getByTestId("suggestion");
    await user.click(suggestion);

    expect(onClick).toHaveBeenCalledWith("a long text value");
  });
});
