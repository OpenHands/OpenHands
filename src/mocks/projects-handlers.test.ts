import { afterEach, describe, expect, it } from "vitest";
import { resetMockProjects } from "#/mocks/projects-handlers";

const API = "http://127.0.0.1:8000/api";

const setProjects = (enabled: boolean) =>
  fetch(`${API}/settings/experimental/projects`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });

afterEach(() => {
  resetMockProjects();
});

describe("Projects MSW handlers", () => {
  it("answers 404 while the feature is off, like the server", async () => {
    const response = await fetch(`${API}/projects`);
    expect(response.status).toBe(404);
  });

  it("serves a board with every kind of card, files and memory", async () => {
    await setProjects(true);
    const { projects } = await (await fetch(`${API}/projects`)).json();
    expect(projects).toHaveLength(1);

    const board = await (
      await fetch(`${API}/projects/${projects[0].id}/conversations`)
    ).json();
    expect(
      new Set(
        board.conversations.map((c: { attention: string }) => c.attention),
      ),
    ).toEqual(
      new Set(["needs_input", "needs_you", "working", "idle", "resolved"]),
    );

    const memory = await (
      await fetch(`${API}/projects/${projects[0].id}/memory`)
    ).json();
    expect(memory.files[0].name).toBe("MEMORY.md");
  });

  it("resolves and reopens a member through the project's routes", async () => {
    await setProjects(true);
    const { projects } = await (await fetch(`${API}/projects`)).json();
    const base = `${API}/projects/${projects[0].id}/conversations`;
    const idle = "04000000000040008000000000000001";

    expect(
      (await fetch(`${base}/${idle}/resolve`, { method: "POST" })).status,
    ).toBe(204);
    let board = await (await fetch(base)).json();
    expect(
      board.conversations.find(
        (c: { title: string }) => c.title === "Install docs",
      ).attention,
    ).toBe("resolved");

    await fetch(`${base}/${idle}/reopen`, { method: "POST" });
    board = await (await fetch(base)).json();
    expect(
      board.conversations.find(
        (c: { title: string }) => c.title === "Install docs",
      ).attention,
    ).toBe("idle");
  });

  it("renames and deletes a project", async () => {
    await setProjects(true);
    const { projects } = await (await fetch(`${API}/projects`)).json();
    const url = `${API}/projects/${projects[0].id}`;
    const renamed = await (
      await fetch(url, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "Release 0.3" }),
      })
    ).json();
    expect(renamed.name).toBe("Release 0.3");

    await fetch(url, { method: "DELETE" });
    expect((await fetch(url)).status).toBe(404);
  });
});
