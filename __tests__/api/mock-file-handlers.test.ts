import { describe, expect, it } from "vitest";

describe("mock file handlers", () => {
  it("returns home directory metadata on the direct agent-server path", async () => {
    const response = await fetch("http://localhost:3000/api/file/home");
    const body = (await response.json()) as {
      home: string;
      favorites: { label: string; path: string }[];
    };

    expect(response.status).toBe(200);
    expect(body.home).toBe("/home/openhands");
    expect(body.favorites).toEqual([
      { label: "Downloads", path: "/home/openhands/Downloads" },
    ]);
  });

  it("returns subdirectories on the direct agent-server path", async () => {
    const response = await fetch(
      "http://localhost:3000/api/file/search_subdirs?path=%2Fprojects",
    );
    const body = (await response.json()) as {
      path: string;
      items: { name: string; path: string }[];
      subdirs: { name: string; path: string }[];
      next_page_id: string | null;
    };

    expect(response.status).toBe(200);
    // In test mode shouldReturnProjectMock is false, so items is empty
    expect(body.items).toEqual([]);
    expect(body.subdirs).toEqual([]);
    expect(body.next_page_id).toBeNull();
  });
});
