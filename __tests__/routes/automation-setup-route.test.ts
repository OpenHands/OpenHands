import { describe, expect, it } from "vitest";
import routes from "#/routes";

describe("automation setup route", () => {
  it("claims the path an automation card links to", () => {
    // Arrange — a card that ships a setup experience links to
    // `/automations/new/<id>`, so a route has to serve that shape.
    const rootLayout = routes.find(
      (route) => route.file === "routes/root-layout.tsx",
    );

    // Act
    const setupRoute = rootLayout?.children?.find(
      (route) => route.file === "routes/automation-setup-route.tsx",
    );

    // Assert
    expect(setupRoute?.path).toBe("automations/new/:automationId");
  });
});
