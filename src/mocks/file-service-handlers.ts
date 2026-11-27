import { delay, http, HttpResponse } from "msw";

export const FILE_VARIANTS_1 = ["file1.txt", "file2.txt", "file3.txt"];
export const FILE_VARIANTS_2 = [
  "reboot_skynet.exe",
  "target_list.txt",
  "terminator_blueprint.txt",
];

const MOCK_FILE_BROWSER_HOME = {
  home: "/home/openhands",
  favorites: [{ label: "Downloads", path: "/home/openhands/Downloads" }],
  locations: [],
};

const MOCK_SUBDIRECTORIES_BY_PATH: Record<
  string,
  { name: string; path: string }[]
> = {
  "/projects": [
    { name: "demo-app", path: "/projects/demo-app" },
    { name: "sample-tools", path: "/projects/sample-tools" },
    { name: "notes-service", path: "/projects/notes-service" },
  ],
  "/projects/demo-app": [
    {
      name: "web-client",
      path: "/projects/demo-app/web-client",
    },
    {
      name: "api-service",
      path: "/projects/demo-app/api-service",
    },
  ],
  "/projects/demo-app/web-client": [],
};

const shouldReturnProjectMock =
  typeof import.meta.env.MODE === "string" && import.meta.env.MODE !== "test";

export const FILE_SERVICE_HANDLERS = [
  http.all("*/api/bash/execute_bash_command", async () =>
    HttpResponse.json({
      command: "",
      exit_code: 0,
      output: "",
    }),
  ),

  http.get("*/api/file/home", async () =>
    HttpResponse.json(MOCK_FILE_BROWSER_HOME),
  ),

  http.get("*/api/file/search_subdirs", async ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get("path") ?? "";
    const items = shouldReturnProjectMock
      ? (MOCK_SUBDIRECTORIES_BY_PATH[path] ?? [])
      : [];

    return HttpResponse.json({
      path,
      items,
      subdirs: items,
      next_page_id: null,
    });
  }),

  http.get("*/api/file/:path", async ({ params }) =>
    HttpResponse.json({
      path: `/${params.path?.toString() ?? "home"}`,
      subdirs: [],
    }),
  ),

  http.get(
    "/api/conversations/:conversationId/list-files",
    async ({ params }) => {
      await delay();

      const cid = params.conversationId?.toString();
      if (!cid) return HttpResponse.json(null, { status: 400 });

      return cid === "test-conversation-id-2"
        ? HttpResponse.json(FILE_VARIANTS_2)
        : HttpResponse.json(FILE_VARIANTS_1);
    },
  ),

  http.get(
    "/api/conversations/:conversationId/select-file",
    async ({ request }) => {
      await delay();

      const url = new URL(request.url);
      const file = url.searchParams.get("file")?.toString();
      if (file) {
        return HttpResponse.json({ code: `Content of ${file}` });
      }

      return HttpResponse.json(null, { status: 404 });
    },
  ),
];
