import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ClerkGate } from "#/components/features/auth/clerk-gate";
import {
  getClerkAllowedEmailDomains,
  isClerkEnabled,
  isEmailDomainAllowed,
} from "#/api/clerk-config";

// The gate is the only thing under test; Clerk's own components are
// replaced with the smallest stand-ins that let us drive each branch.
const clerkState = vi.hoisted(() => ({
  isSignedIn: false,
  user: null as null | {
    primaryEmailAddress: {
      emailAddress: string;
      verification: { status: string };
    };
  },
}));

vi.mock("@clerk/clerk-react", () => ({
  ClerkProvider: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="clerk-provider">{children}</div>
  ),
  ClerkLoading: () => null,
  ClerkLoaded: ({ children }: { children: React.ReactNode }) => children,
  SignedIn: ({ children }: { children: React.ReactNode }) =>
    clerkState.isSignedIn ? children : null,
  SignedOut: ({ children }: { children: React.ReactNode }) =>
    clerkState.isSignedIn ? null : children,
  SignIn: () => <div data-testid="clerk-sign-in-widget" />,
  SignOutButton: ({ children }: { children: React.ReactNode }) => children,
  useUser: () => ({ isLoaded: true, user: clerkState.user }),
}));

const CANVAS_TEST_ID = "canvas-children";

function renderGate() {
  return render(
    <ClerkGate>
      <div data-testid={CANVAS_TEST_ID} />
    </ClerkGate>,
  );
}

function signIn(email: string, status = "verified") {
  clerkState.isSignedIn = true;
  clerkState.user = {
    primaryEmailAddress: { emailAddress: email, verification: { status } },
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
  clerkState.isSignedIn = false;
  clerkState.user = null;
});

describe("clerk-config", () => {
  it("is disabled when no publishable key is configured", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "");
    expect(isClerkEnabled()).toBe(false);
  });

  it("is enabled once a publishable key is configured", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    expect(isClerkEnabled()).toBe(true);
  });

  it("parses the allowlist, tolerating spacing, case, and a leading @", () => {
    vi.stubEnv("VITE_CLERK_ALLOWED_EMAIL_DOMAINS", " Acme.com , @acme.dev ");
    expect(getClerkAllowedEmailDomains()).toEqual(["acme.com", "acme.dev"]);
  });

  it("allows every address when the allowlist is empty", () => {
    expect(isEmailDomainAllowed("anyone@example.com", [])).toBe(true);
  });

  it("matches the domain, not a substring of it", () => {
    expect(isEmailDomainAllowed("dev@acme.com", ["acme.com"])).toBe(true);
    // `notacme.com` ends with `acme.com`; a substring check would pass it.
    expect(isEmailDomainAllowed("dev@notacme.com", ["acme.com"])).toBe(false);
  });

  it("rejects a missing address when an allowlist is set", () => {
    expect(isEmailDomainAllowed(null, ["acme.com"])).toBe(false);
  });
});

describe("ClerkGate", () => {
  it("renders children untouched when Clerk is not configured", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "");
    renderGate();

    expect(screen.getByTestId(CANVAS_TEST_ID)).toBeInTheDocument();
    expect(screen.queryByTestId("clerk-provider")).not.toBeInTheDocument();
  });

  it("shows the sign-in screen instead of the canvas when signed out", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    renderGate();

    expect(screen.getByTestId("clerk-sign-in")).toBeInTheDocument();
    expect(screen.queryByTestId(CANVAS_TEST_ID)).not.toBeInTheDocument();
  });

  it("renders the canvas for a signed-in user when no allowlist is set", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    signIn("dev@anywhere.com");
    renderGate();

    expect(screen.getByTestId(CANVAS_TEST_ID)).toBeInTheDocument();
  });

  it("renders the canvas for an allowlisted signed-in user", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    vi.stubEnv("VITE_CLERK_ALLOWED_EMAIL_DOMAINS", "acme.com");
    signIn("dev@acme.com");
    renderGate();

    expect(screen.getByTestId(CANVAS_TEST_ID)).toBeInTheDocument();
  });

  it("blocks a signed-in user whose domain is not allowlisted", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    vi.stubEnv("VITE_CLERK_ALLOWED_EMAIL_DOMAINS", "acme.com");
    signIn("stranger@example.com");
    renderGate();

    expect(screen.getByTestId("clerk-not-on-team")).toBeInTheDocument();
    expect(screen.queryByTestId(CANVAS_TEST_ID)).not.toBeInTheDocument();
  });

  it("blocks an allowlisted domain on an unverified address", () => {
    vi.stubEnv("VITE_CLERK_PUBLISHABLE_KEY", "pk_test_abc");
    vi.stubEnv("VITE_CLERK_ALLOWED_EMAIL_DOMAINS", "acme.com");
    signIn("dev@acme.com", "unverified");
    renderGate();

    expect(screen.getByTestId("clerk-not-on-team")).toBeInTheDocument();
    expect(screen.queryByTestId(CANVAS_TEST_ID)).not.toBeInTheDocument();
  });
});
