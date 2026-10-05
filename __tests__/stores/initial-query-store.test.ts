import { beforeEach, describe, expect, it } from "vitest";
import { useInitialQueryStore } from "#/stores/initial-query-store";

const emptyState = {
  files: [],
  initialPrompt: null,
  selectedRepository: null,
  selectedRepositoryProvider: null,
  replayJson: null,
};

describe("useInitialQueryStore", () => {
  beforeEach(() => {
    useInitialQueryStore.getState().reset();
  });

  it("adds, removes, and clears files", () => {
    const store = useInitialQueryStore.getState();
    store.addFile("first");
    store.addFile("second");
    store.addFile("third");
    expect(useInitialQueryStore.getState().files).toEqual([
      "first",
      "second",
      "third",
    ]);

    useInitialQueryStore.getState().removeFile(1);
    expect(useInitialQueryStore.getState().files).toEqual(["first", "third"]);
    useInitialQueryStore.getState().removeFile(99);
    expect(useInitialQueryStore.getState().files).toEqual(["first", "third"]);
    useInitialQueryStore.getState().clearFiles();
    expect(useInitialQueryStore.getState().files).toEqual([]);
  });

  it("sets and clears prompt and repository selections", () => {
    const repository = {
      id: "repository-1",
      full_name: "owner/repository",
      git_provider: "github",
      is_public: true,
    } as const;
    const store = useInitialQueryStore.getState();
    store.setInitialPrompt("Build it");
    store.setSelectedRepository(repository);
    store.setSelectedRepositoryProvider("github");
    store.setReplayJson('{"event":"message"}');
    expect(useInitialQueryStore.getState()).toMatchObject({
      initialPrompt: "Build it",
      selectedRepository: repository,
      selectedRepositoryProvider: "github",
      replayJson: '{"event":"message"}',
    });

    useInitialQueryStore.getState().clearInitialPrompt();
    useInitialQueryStore.getState().clearSelectedRepository();
    useInitialQueryStore.getState().setSelectedRepositoryProvider(null);
    useInitialQueryStore.getState().setReplayJson(null);
    expect(useInitialQueryStore.getState()).toMatchObject(emptyState);
  });

  it("resets every field after populated state", () => {
    const store = useInitialQueryStore.getState();
    store.addFile("image");
    store.setInitialPrompt("Prompt");
    store.setSelectedRepositoryProvider("gitlab");
    store.setReplayJson("{}");
    useInitialQueryStore.getState().reset();
    expect(useInitialQueryStore.getState()).toMatchObject(emptyState);
  });
});
