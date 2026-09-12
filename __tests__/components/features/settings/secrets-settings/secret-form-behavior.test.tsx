import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CustomSecretWithoutValue } from "#/api/secrets-service.types";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import { I18nKey } from "#/i18n/declaration";

const mocks = vi.hoisted(() => ({
  secrets: undefined as CustomSecretWithoutValue[] | undefined,
  create: vi.fn(),
  update: vi.fn(),
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `wrong-namespace:${key}`,
  }),
}));

vi.mock("#/hooks/query/use-get-secrets", () => ({
  useSearchSecrets: () => ({ data: mocks.secrets }),
}));

vi.mock("#/hooks/mutation/use-create-secret", () => ({
  useCreateSecret: () => ({ mutate: mocks.create }),
}));

vi.mock("#/hooks/mutation/use-update-secret", () => ({
  useUpdateSecret: () => ({ mutate: mocks.update }),
}));

interface RenderFormOptions {
  mode?: "add" | "edit";
  selectedSecret?: string | null;
  onCancel?: () => void;
}

function renderForm({
  mode = "add",
  selectedSecret = null,
  onCancel = vi.fn(),
}: RenderFormOptions = {}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const invalidate = vi.spyOn(queryClient, "invalidateQueries");
  const view = render(
    <QueryClientProvider client={queryClient}>
      <SecretForm
        mode={mode}
        selectedSecret={selectedSecret}
        onCancel={onCancel}
      />
    </QueryClientProvider>,
  );
  return { ...view, invalidate, onCancel };
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.secrets = undefined;
});

describe("Secret form behavior", () => {
  it("renders the add form constraints and cancels without saving", async () => {
    const user = userEvent.setup();
    const { onCancel } = renderForm();

    expect(screen.getByTestId("add-secret-form")).toBeInTheDocument();
    expect(screen.getByTestId("name-input")).toHaveAttribute(
      "pattern",
      "^[a-zA-Z][a-zA-Z0-9_]{0,63}$",
    );
    expect(screen.getByTestId("name-input")).toBeRequired();
    expect(screen.getByTestId("value-input")).toBeRequired();
    expect(screen.getByTestId("value-input")).toHaveClass(
      "resize-none",
      "placeholder:italic",
      "disabled:bg-[var(--oh-surface-raised)]",
      "disabled:border-[var(--oh-border-subtle)]",
      "disabled:cursor-not-allowed",
    );
    expect(screen.getByTestId("description-input")).toHaveValue("");
    expect(screen.getByTestId("description-input")).toHaveClass(
      "disabled:bg-[var(--oh-surface-raised)]",
      "disabled:border-[var(--oh-border-subtle)]",
    );
    expect(screen.getByTestId("submit-button")).toHaveTextContent(
      I18nKey.SECRETS$ADD_SECRET,
    );
    expect(screen.getByTestId("submit-button")).not.toHaveTextContent(
      I18nKey.SECRETS$EDIT_SECRET,
    );

    await user.type(screen.getByTestId("name-input"), "CANCELLED_SECRET");
    await user.type(screen.getByTestId("value-input"), "unused-value");

    await user.click(screen.getByTestId("cancel-button"));
    expect(onCancel).toHaveBeenCalledOnce();
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("renders an edit form with the selected secret and trimmed description", () => {
    mocks.secrets = [
      { name: "EDIT_ME", description: "  production credential  " },
    ];
    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });

    expect(screen.getByTestId("edit-secret-form")).toBeInTheDocument();
    expect(screen.getByTestId("name-input")).toHaveValue("EDIT_ME");
    // Edit mode now exposes an optional value field that starts blank; leaving
    // it blank keeps the existing secret value.
    expect(screen.getByTestId("value-input")).toHaveValue("");
    expect(screen.getByTestId("description-input")).toHaveValue(
      "production credential",
    );
    expect(screen.getByTestId("submit-button")).toHaveTextContent(
      I18nKey.SECRETS$EDIT_SECRET,
    );
    expect(screen.getByTestId("submit-button")).not.toHaveTextContent(
      I18nKey.SECRETS$ADD_SECRET,
    );
  });

  it("ignores a selected secret while rendering the add form", () => {
    mocks.secrets = [
      { name: "EDIT_ME", description: "must not prefill the add form" },
    ];

    renderForm({ mode: "add", selectedSecret: "EDIT_ME" });

    expect(screen.getByTestId("name-input")).toHaveValue("");
    expect(screen.getByTestId("description-input")).toHaveValue("");
  });

  it("leaves the edit description blank when metadata is absent or unmatched", () => {
    const first = renderForm({ mode: "edit", selectedSecret: "MISSING" });
    expect(screen.getByTestId("description-input")).toHaveValue("");
    first.unmount();

    mocks.secrets = [{ name: "MISSING" }];
    const second = renderForm({ mode: "edit", selectedSecret: "MISSING" });
    expect(screen.getByTestId("description-input")).toHaveValue("");
    second.unmount();

    mocks.secrets = [{ name: "OTHER", description: "other description" }];
    const third = renderForm({ mode: "edit", selectedSecret: "MISSING" });
    expect(screen.getByTestId("description-input")).toHaveValue("");
    third.unmount();

    renderForm({ mode: "edit", selectedSecret: null });
    expect(screen.getByTestId("name-input")).toHaveValue("");
    expect(screen.getByTestId("description-input")).toHaveValue("");
  });

  it("does nothing when a submitted form has no secret name", () => {
    renderForm();

    fireEvent.submit(screen.getByTestId("add-secret-form"));

    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.update).not.toHaveBeenCalled();
    expect(
      screen.queryByText(I18nKey.SECRETS$SECRET_VALUE_REQUIRED),
    ).not.toBeInTheDocument();
  });

  it("does nothing when the secret name control is absent", () => {
    renderForm();
    screen.getByTestId("name-input").remove();

    const errors: unknown[] = [];
    const captureError = (event: ErrorEvent) => {
      event.preventDefault();
      errors.push(event.error);
    };
    window.addEventListener("error", captureError);

    try {
      fireEvent.submit(screen.getByTestId("add-secret-form"));
    } finally {
      window.removeEventListener("error", captureError);
    }

    expect(errors).toHaveLength(0);
    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.update).not.toHaveBeenCalled();
  });

  it("creates safely when its optional description control is missing", () => {
    renderForm();
    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "NO_DESCRIPTION_CONTROL" },
    });
    fireEvent.change(screen.getByTestId("value-input"), {
      target: { value: "secret-value" },
    });
    screen.getByTestId("description-input").remove();

    fireEvent.submit(screen.getByTestId("add-secret-form"));

    expect(mocks.create).toHaveBeenCalledWith(
      {
        name: "NO_DESCRIPTION_CONTROL",
        value: "secret-value",
        description: undefined,
      },
      expect.any(Object),
    );
  });

  it("rejects an add when the secret name is already used", async () => {
    const user = userEvent.setup();
    mocks.secrets = [
      { name: "OTHER_SECRET", description: "existing" },
      { name: "DUPLICATE", description: "existing" },
    ];
    renderForm();

    await user.type(screen.getByTestId("name-input"), "DUPLICATE");
    await user.type(screen.getByTestId("value-input"), "secret-value");
    await user.click(screen.getByTestId("submit-button"));

    expect(screen.getByText(I18nKey.SECRETS$SECRET_ALREADY_EXISTS)).toHaveClass(
      "text-red-500",
      "text-sm",
    );
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("requires a non-whitespace value when adding a secret", () => {
    renderForm();
    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "NEW_SECRET" },
    });
    fireEvent.change(screen.getByTestId("value-input"), {
      target: { value: "   " },
    });

    fireEvent.submit(screen.getByTestId("add-secret-form"));

    expect(
      screen.getByText(I18nKey.SECRETS$SECRET_VALUE_REQUIRED),
    ).toBeInTheDocument();
    expect(mocks.create).not.toHaveBeenCalled();
  });

  it("creates a trimmed secret, refreshes both secret lists, and closes after settlement", async () => {
    const user = userEvent.setup();
    const { invalidate, onCancel } = renderForm();

    await user.type(screen.getByTestId("name-input"), "NEW_SECRET");
    await user.type(screen.getByTestId("value-input"), "  secret value  ");
    await user.type(screen.getByTestId("description-input"), "for deployments");
    await user.click(screen.getByTestId("submit-button"));

    expect(mocks.create).toHaveBeenCalledWith(
      {
        name: "NEW_SECRET",
        value: "secret value",
        description: "for deployments",
      },
      expect.any(Object),
    );
    const options = mocks.create.mock.calls[0][1];

    expect(invalidate).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();

    await act(async () => options.onSuccess());

    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["secrets-search"] });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["secrets"] });
    expect(onCancel).not.toHaveBeenCalled();

    options.onSettled();

    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("omits an empty optional description when creating", async () => {
    const user = userEvent.setup();
    renderForm();

    await user.type(screen.getByTestId("name-input"), "NO_DESCRIPTION");
    await user.type(screen.getByTestId("value-input"), "value");
    await user.click(screen.getByTestId("submit-button"));

    expect(mocks.create).toHaveBeenCalledWith(
      { name: "NO_DESCRIPTION", value: "value", description: undefined },
      expect.any(Object),
    );
  });

  it("edits the selected secret without treating its unchanged name as a duplicate", async () => {
    const user = userEvent.setup();
    mocks.secrets = [
      { name: "EDIT_ME", description: "old description" },
      { name: "OTHER_SECRET" },
    ];
    const { invalidate, onCancel } = renderForm({
      mode: "edit",
      selectedSecret: "EDIT_ME",
    });

    await user.clear(screen.getByTestId("description-input"));
    await user.type(screen.getByTestId("description-input"), "new description");
    await user.click(screen.getByTestId("submit-button"));

    expect(mocks.update).toHaveBeenCalledWith(
      {
        secretToEdit: "EDIT_ME",
        name: "EDIT_ME",
        description: "new description",
      },
      expect.any(Object),
    );
    const options = mocks.update.mock.calls[0][1];

    expect(invalidate).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();

    await act(async () => options.onSuccess());

    expect(invalidate).toHaveBeenCalledTimes(2);
    expect(onCancel).not.toHaveBeenCalled();

    options.onSettled();

    expect(onCancel).toHaveBeenCalledOnce();
  });

  it("rejects renaming an edited secret to an existing name", async () => {
    const user = userEvent.setup();
    mocks.secrets = [{ name: "EDIT_ME" }, { name: "OTHER_SECRET" }];
    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });

    await user.clear(screen.getByTestId("name-input"));
    await user.type(screen.getByTestId("name-input"), "OTHER_SECRET");
    await user.click(screen.getByTestId("submit-button"));

    expect(
      screen.getByText(I18nKey.SECRETS$SECRET_ALREADY_EXISTS),
    ).toBeInTheDocument();
    expect(mocks.update).not.toHaveBeenCalled();
  });

  it("omits an empty edit description and does not edit without a selected secret", async () => {
    const user = userEvent.setup();
    mocks.secrets = [{ name: "EDIT_ME" }];
    const selected = renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });
    // The edit submit is enabled only once the form is dirty; typing a
    // replacement value enables it while the description stays blank, so the
    // empty description must still be omitted from the update payload.
    await user.type(screen.getByTestId("value-input"), "new-value");
    await user.click(screen.getByTestId("submit-button"));
    expect(mocks.update).toHaveBeenCalledWith(
      {
        secretToEdit: "EDIT_ME",
        name: "EDIT_ME",
        description: undefined,
        value: "new-value",
      },
      expect.any(Object),
    );
    selected.unmount();

    mocks.update.mockClear();
    renderForm({ mode: "edit", selectedSecret: null });
    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "ORPHANED_EDIT" },
    });
    fireEvent.submit(screen.getByTestId("edit-secret-form"));
    expect(mocks.update).not.toHaveBeenCalled();
  });

  it("keeps the add submit disabled until both name and value are present", async () => {
    const user = userEvent.setup();
    renderForm({ mode: "add" });
    const submit = screen.getByTestId("submit-button");

    expect(submit).toBeDisabled();
    await user.type(screen.getByTestId("name-input"), "NEW_SECRET");
    expect(submit).toBeDisabled();
    await user.type(screen.getByTestId("value-input"), "a-value");
    expect(submit).toBeEnabled();

    // Removing the value disables it again.
    await user.clear(screen.getByTestId("value-input"));
    expect(submit).toBeDisabled();
  });

  it("labels the value field differently for add and edit modes", () => {
    mocks.secrets = [{ name: "EDIT_ME" }];
    const { unmount } = renderForm({ mode: "add" });
    expect(screen.getByText(I18nKey.FORM$VALUE)).toBeInTheDocument();
    expect(
      screen.queryByText(I18nKey.SECRETS$SECRET_VALUE_LEAVE_BLANK),
    ).not.toBeInTheDocument();
    unmount();

    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });
    expect(
      screen.getByText(I18nKey.SECRETS$SECRET_VALUE_LEAVE_BLANK),
    ).toBeInTheDocument();
    expect(screen.queryByText(I18nKey.FORM$VALUE)).not.toBeInTheDocument();
  });

  it("enables the edit submit only when a field actually changes", async () => {
    const user = userEvent.setup();
    mocks.secrets = [{ name: "EDIT_ME", description: "orig" }];
    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });
    const submit = screen.getByTestId("submit-button");

    // Unchanged form: disabled.
    expect(submit).toBeDisabled();

    // Renaming enables it; reverting to the original name disables it again.
    await user.clear(screen.getByTestId("name-input"));
    await user.type(screen.getByTestId("name-input"), "RENAMED");
    expect(submit).toBeEnabled();
    await user.clear(screen.getByTestId("name-input"));
    await user.type(screen.getByTestId("name-input"), "EDIT_ME");
    expect(submit).toBeDisabled();

    // Editing the description enables it.
    await user.type(screen.getByTestId("description-input"), " updated");
    expect(submit).toBeEnabled();
  });

  it("edits only the description while keeping the existing value when value is blank", async () => {
    const user = userEvent.setup();
    mocks.secrets = [{ name: "EDIT_ME", description: "orig" }];
    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });

    await user.type(screen.getByTestId("description-input"), " changed");
    await user.click(screen.getByTestId("submit-button"));

    expect(mocks.update).toHaveBeenCalledWith(
      {
        secretToEdit: "EDIT_ME",
        name: "EDIT_ME",
        description: "orig changed",
        value: undefined,
      },
      expect.any(Object),
    );
  });

  it("requires a non-whitespace value to enable the add submit", async () => {
    const user = userEvent.setup();
    renderForm({ mode: "add" });
    await user.type(screen.getByTestId("name-input"), "NEW_SECRET");
    await user.type(screen.getByTestId("value-input"), "    ");
    expect(screen.getByTestId("submit-button")).toBeDisabled();
  });

  it("ignores whitespace-only edits when deciding if the edit form is dirty", async () => {
    const user = userEvent.setup();
    mocks.secrets = [{ name: "EDIT_ME", description: "orig" }];
    renderForm({ mode: "edit", selectedSecret: "EDIT_ME" });
    const submit = screen.getByTestId("submit-button");

    // Padding the name with surrounding whitespace is not a real change.
    await user.type(screen.getByTestId("name-input"), "   ");
    expect(submit).toBeDisabled();
    await user.clear(screen.getByTestId("name-input"));
    await user.type(screen.getByTestId("name-input"), "EDIT_ME");

    // A whitespace-only value is treated as blank (keep existing).
    await user.type(screen.getByTestId("value-input"), "   ");
    expect(submit).toBeDisabled();
  });

  it("re-prefills the fields when the selected secret changes", () => {
    mocks.secrets = [
      { name: "FIRST", description: "  one  " },
      { name: "SECOND", description: "two" },
    ];
    const { rerender } = renderForm({ mode: "edit", selectedSecret: "FIRST" });
    expect(screen.getByTestId("name-input")).toHaveValue("FIRST");
    expect(screen.getByTestId("description-input")).toHaveValue("one");

    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <SecretForm mode="edit" selectedSecret="SECOND" onCancel={vi.fn()} />
      </QueryClientProvider>,
    );
    expect(screen.getByTestId("name-input")).toHaveValue("SECOND");
    expect(screen.getByTestId("description-input")).toHaveValue("two");
  });
});
