import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EditableSourceView } from "#/components/features/files-tab/editable-source-view";

const mutateAsyncMock = vi.fn();

vi.mock("#/hooks/mutation/use-save-workspace-file", () => ({
  useSaveWorkspaceFile: () => ({
    mutateAsync: mutateAsyncMock,
    isPending: false,
  }),
}));

vi.mock("@monaco-editor/react", () => ({
  Editor: (props: {
    value?: string;
    onChange?: (value: string | undefined) => void;
  }) => (
    <textarea
      data-testid="monaco-editor-mock"
      value={props.value}
      onChange={(event) => props.onChange?.(event.target.value)}
    />
  ),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

function renderView(text = "print('hello')\n") {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <EditableSourceView path="debug_shuffle.py" text={text} />
    </QueryClientProvider>,
  );
}

describe("EditableSourceView", () => {
  beforeEach(() => {
    mutateAsyncMock.mockReset();
    mutateAsyncMock.mockResolvedValue(undefined);
  });

  it("enables save after the content changes and persists via the mutation", async () => {
    const user = userEvent.setup();
    renderView();

    const save = screen.getByTestId("editable-source-save");
    expect(save).toBeDisabled();

    const editor = screen.getByTestId("monaco-editor-mock");
    await user.clear(editor);
    await user.type(editor, "print edited");

    expect(screen.getByTestId("editable-source-view")).toHaveAttribute(
      "data-dirty",
      "true",
    );
    expect(save).toBeEnabled();

    await user.click(save);

    await waitFor(() => {
      expect(mutateAsyncMock).toHaveBeenCalledWith({
        relativePath: "debug_shuffle.py",
        content: "print edited",
      });
    });
  });

  it("discards unsaved edits back to the baseline", async () => {
    const user = userEvent.setup();
    renderView("original\n");

    const editor = screen.getByTestId("monaco-editor-mock");
    await user.clear(editor);
    await user.type(editor, "changed");
    expect(editor).toHaveValue("changed");

    await user.click(screen.getByTestId("editable-source-discard"));
    expect(editor).toHaveValue("original\n");
    expect(screen.getByTestId("editable-source-view")).toHaveAttribute(
      "data-dirty",
      "false",
    );
  });
});
