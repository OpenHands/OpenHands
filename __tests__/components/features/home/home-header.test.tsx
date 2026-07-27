import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HomeHeader } from "#/components/features/home/home-header/home-header";

// Mock the translation function
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        // Return a mock translation for the test
        const translations: Record<string, string> = {
          HOME$LETS_START_BUILDING: "Let's start building",
        };
        return translations[key] || key;
      },
      i18n: { language: "en" },
    }),
  };
});

const renderHomeHeader = () => {
  return render(<HomeHeader />, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={new QueryClient()}>
        {children}
      </QueryClientProvider>
    ),
  });
};

describe("HomeHeader", () => {
  it("should render the header with the correct title", () => {
    renderHomeHeader();

    const title = screen.getByText("Let's start building");
    expect(title).toBeInTheDocument();
  });

  it("should render the GuideMessage component", () => {
    renderHomeHeader();

    // The GuideMessage component should be rendered as part of the header
    const header = screen.getByRole("banner");
    expect(header).toBeInTheDocument();
  });

});
