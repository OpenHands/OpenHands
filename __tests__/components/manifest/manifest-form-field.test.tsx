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
import type { SetupFormField as SetupFormFieldDefinition } from "#/manifests/types";

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
function Harness({ onValueChange }: { onValueChange: (value: string) => void }) {
  const [value, setValue] = useState("");

  return (
    <SetupFormField
      name="repository"
      field={REPOSITORY_FIELD}
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

function renderRepositoryField(backend: Backend) {
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
        <Harness onValueChange={onValueChange} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );

  return { onValueChange, user: userEvent.setup() };
}

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

  it("browses the account's repositories on a cloud backend", () => {
    // Arrange / Act
    renderRepositoryField(CLOUD_BACKEND);

    // Assert — the picker stays the way a repository is chosen wherever it can
    // actually answer, so there is nothing to type into.
    expect(screen.getByTestId("git-repo-dropdown")).toBeInTheDocument();
    expect(screen.queryByTestId("setup-field-repository")).toBeNull();
  });
});
