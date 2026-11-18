import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createRoutesStub } from "react-router";
import { ErrorBoundary } from "#/routes/root-layout";
import { I18nKey } from "#/i18n/declaration";

describe("root layout error boundary translation protection (#19755)", () => {
  it("renders ErrorShell with translate='no' to guard against translation loops", async () => {
    const ErrorRouterStub = createRoutesStub([
      {
        path: "/",
        Component: () => null,
        ErrorBoundary,
        loader: () => {
          throw new Error("Translation protection test");
        },
      },
    ]);

    const { container } = render(<ErrorRouterStub initialEntries={["/"]} />);

    expect(
      await screen.findByRole("heading", { name: I18nKey.ERROR$GENERIC }),
    ).toBeInTheDocument();

    const errorShell = container.querySelector("[translate='no']");
    expect(errorShell).toBeInTheDocument();
    expect(errorShell).toHaveAttribute("translate", "no");
  });
});
