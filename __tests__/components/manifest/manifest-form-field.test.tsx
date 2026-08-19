import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { SetupFormField } from "#/components/features/manifest/manifest-form-field";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Backend } from "#/api/backend-registry/types";
import type {
  SetupFormField as SetupFormFieldDefinition,
  SetupFormValue,
} from "#/manifests/types";

const LOCAL_BACKEND: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:18000",
  apiKey: "",
  kind: "local",
};

const CLOUD_BACKEND: Backend = {
  id: "cloud-1",
  name: "OpenHands Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

/** The repository input `github-pr-reviewer` declares, in its published shape. */
const REPOSITORY_FIELD: SetupFormFieldDefinition = {
  type: "repo-picker",
  label: "Repository",
  help: "The repository whose pull requests will be reviewed.",
  provider: "github",
  required: true,
};

/** Holds the field value the way the setup dialog does, so typing accumulates. */
function Harness({
  field = REPOSITORY_FIELD,
  initialValue = "",
  onValueChange,
}: {
  field?: SetupFormFieldDefinition;
  initialValue?: SetupFormValue;
  onValueChange: (value: SetupFormValue) => void;
}) {
  const [value, setValue] = useState<SetupFormValue>(initialValue);

  return (
    <SetupFormField
      name="repository"
      field={field}
      value={value}
      options={[]}
      repository={null}
      disabled={false}
      onChange={(next) => {
        setValue(next);
        onValueChange(next);
      }}
      onRepositoryChange={vi.fn()}
      onBlur={vi.fn()}
    />
  );
}

function renderRepositoryField(
  backend: Backend,
  harness: {
    field?: SetupFormFieldDefinition;
    initialValue?: SetupFormValue;
  } = {},
) {
  setRegisteredBackends([backend]);
  setActiveSelection({ backendId: backend.id });

  const onValueChange = vi.fn();
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      <ActiveBackendProvider>
        <Harness {...harness} onValueChange={onValueChange} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );

  return { onValueChange, user: userEvent.setup() };
}

/** The same field once the entry asks for several repositories. */
const REPOSITORIES_FIELD: SetupFormFieldDefinition = {
  ...REPOSITORY_FIELD,
  label: "Repositories",
  multiple: true,
};

beforeEach(() => {
  __resetActiveStoreForTests();
});

afterEach(() => {
  __resetActiveStoreForTests();
});

describe("SetupFormField repo-picker", () => {
  it("lets a repository be typed on a backend that cannot list them", async () => {
    // Arrange — a local backend, where GitService answers every repository
    // query with an empty page.
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND);

    // Act
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/agent-server-gui",
    );

    // Assert — the required field is answerable, in the `owner/repo` shape the
    // create payload sends as `repos[0].url`.
    expect(onValueChange).toHaveBeenLastCalledWith(
      "OpenHands/agent-server-gui",
    );
  });

  it("collects several repositories when the entry asks for several", async () => {
    // Arrange
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: [],
    });

    // Act
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/automation",
    );
    await user.click(screen.getByTestId("setup-list-repository-add"));
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/extensions",
    );
    await user.click(screen.getByTestId("setup-list-repository-add"));

    // Assert — one automation polling both, which is what the entry supports.
    expect(onValueChange).toHaveBeenLastCalledWith([
      "OpenHands/automation",
      "OpenHands/extensions",
    ]);
  });

  it("adds a repository on Enter rather than submitting a half-built list", async () => {
    // Arrange
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: [],
    });

    // Act
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/automation{Enter}",
    );

    // Assert
    expect(onValueChange).toHaveBeenLastCalledWith(["OpenHands/automation"]);
  });

  it("does not add a repository already in the list", async () => {
    // Arrange — adding it twice polls it twice per run for one result.
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: ["OpenHands/automation"],
    });

    // Act
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/automation{Enter}",
    );

    // Assert
    expect(onValueChange).not.toHaveBeenCalled();
  });

  it("removes a repository from the list", async () => {
    // Arrange
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: ["OpenHands/automation", "OpenHands/extensions"],
    });

    // Act
    await user.click(
      screen.getByTestId("setup-list-repository-remove-OpenHands/automation"),
    );

    // Assert
    expect(onValueChange).toHaveBeenLastCalledWith(["OpenHands/extensions"]);
  });

  it("names the input the entry's own label for a screen reader", () => {
    // Arrange — the label is rendered above the list rather than on the input,
    // which is how an input ends up announced as nothing at all.
    renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: [],
    });

    // Assert
    expect(screen.getByRole("textbox", { name: "Repositories" })).toBe(
      screen.getByTestId("setup-field-repository"),
    );
  });

  it("keeps a repository typed but not added, rather than dropping it", async () => {
    // Arrange — the input still shows the text, so leaving the field is the
    // user saying they answered it.
    const { onValueChange, user } = renderRepositoryField(LOCAL_BACKEND, {
      field: REPOSITORIES_FIELD,
      initialValue: ["OpenHands/automation"],
    });

    // Act
    await user.type(
      screen.getByTestId("setup-field-repository"),
      "OpenHands/extensions",
    );
    await user.tab();

    // Assert
    expect(onValueChange).toHaveBeenLastCalledWith([
      "OpenHands/automation",
      "OpenHands/extensions",
    ]);
  });

  it("browses the account's repositories on a cloud backend", () => {
    // Arrange / Act
    renderRepositoryField(CLOUD_BACKEND);

    // Assert — the picker stays the way a repository is chosen wherever it can
    // actually answer, so there is nothing to type into.
    expect(screen.getByTestId("git-repo-dropdown")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-field-repository")).toBeNull();
  });
});
