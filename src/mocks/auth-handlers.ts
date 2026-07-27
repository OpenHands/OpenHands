import { http, HttpResponse } from "msw";
import { GitUser } from "#/types/git";

export const AUTH_HANDLERS = [
  // Cloud proxy is not available in mock mode; return a clean 503 so callers
  // (useAllCloudOrganizations, useBackendsHealth cloud path) fail gracefully
  // instead of falling through to the real dev server and getting a 502.
  http.post("*/api/cloud-proxy", () =>
    HttpResponse.json(
      { error: "cloud proxy not available in mock mode" },
      { status: 503 },
    ),
  ),

  http.get("*/api/user/info", () => {
    const user: GitUser = {
      id: "1",
      login: "octocat",
      avatar_url: "https://avatars.githubusercontent.com/u/583231?v=4",
      company: "GitHub",
      email: "placeholder@placeholder.placeholder",
      name: "monalisa octocat",
    };

    return HttpResponse.json(user);
  }),

  http.post("*/api/authenticate", async () =>
    HttpResponse.json({ message: "Authenticated" }),
  ),

  http.post("*/api/logout", () => HttpResponse.json(null, { status: 200 })),
];
