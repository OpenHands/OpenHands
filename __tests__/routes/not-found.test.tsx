import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { matchRoutes, type RouteObject } from "react-router";
import routes from "#/routes";
import NotFoundRoute from "#/routes/not-found";
import { renderWithProviders } from "../../test-utils";

function matchedRouteFiles(pathname: string) {
  const matches = matchRoutes(routes as unknown as RouteObject[], pathname);
  return matches?.map((match) => (match.route as { file?: string }).file);
}

describe("unknown URLs", () => {
  it.each(["/this-route-does-not-exist", "/settings/does-not-exist"])(
    "resolves %s to the not-found page inside the app shell",
    (pathname) => {
      expect(matchedRouteFiles(pathname)).toEqual([
        "routes/root-layout.tsx",
        "routes/not-found.tsx",
      ]);
    },
  );

  it("keeps known nested routes on their own pages", () => {
    expect(matchedRouteFiles("/settings/llm")).toEqual([
      "routes/root-layout.tsx",
      "routes/settings.tsx",
      "routes/llm-settings.tsx",
    ]);
  });

  it("renders a not-found message with a link back to home", () => {
    renderWithProviders(<NotFoundRoute />);

    expect(
      screen.getByRole("heading", { name: "NOT_FOUND$TITLE" }),
    ).toBeInTheDocument();
    expect(screen.getByText("NOT_FOUND$MESSAGE")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "BUTTON$HOME" })).toHaveAttribute(
      "href",
      "/",
    );
  });
});
