import { describe, expect, it } from "vitest";
import routes from "#/routes";

describe("root index route", () => {
  it("renders the home screen at the deployment root without redirecting to /conversations", () => {
    const rootLayout = routes.find((route) => route.file === "routes/root-layout.tsx");
    const indexRoute = rootLayout?.children?.find((route) => route.index);

    expect(indexRoute?.file).toBe("routes/index-home.tsx");
  });
});
