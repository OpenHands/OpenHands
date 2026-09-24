import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AcpAuthStatusBanner } from "#/components/features/settings/acp-auth-status-banner";

const PREFIX = "settings-acp-auth";

describe("AcpAuthStatusBanner", () => {
  it("shows the 'signed in' banner when the host-login probe authenticated", () => {
    render(
      <AcpAuthStatusBanner
        status="authenticated"
        isChecking={false}
        providerName="Claude Code"
        testIdPrefix={PREFIX}
      />,
    );
    expect(screen.getByTestId(`${PREFIX}-detected`)).toBeInTheDocument();
    expect(
      screen.queryByTestId(`${PREFIX}-configured`),
    ).not.toBeInTheDocument();
  });

  it("shows the checking spinner while the first probe is in flight", () => {
    render(
      <AcpAuthStatusBanner
        status="unknown"
        isChecking
        providerName="Claude Code"
        testIdPrefix={PREFIX}
      />,
    );
    expect(screen.getByTestId(`${PREFIX}-checking`)).toBeInTheDocument();
  });

  it("shows the 'credentials configured' banner when the probe can't confirm a login but a credential is stored (Docker/cloud)", () => {
    render(
      <AcpAuthStatusBanner
        status="unknown"
        isChecking={false}
        credentialsConfigured
        providerName="Claude Code"
        testIdPrefix={PREFIX}
      />,
    );
    expect(screen.getByTestId(`${PREFIX}-configured`)).toBeInTheDocument();
    // Honesty guard: a stored credential must NOT render as the "signed in" banner.
    expect(screen.queryByTestId(`${PREFIX}-detected`)).not.toBeInTheDocument();
  });

  it("renders nothing when there is no login signal and no stored credential", () => {
    const { container } = render(
      <AcpAuthStatusBanner
        status="unknown"
        isChecking={false}
        providerName="Claude Code"
        testIdPrefix={PREFIX}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("prefers the checking spinner over 'configured' while the probe is still in flight", () => {
    render(
      <AcpAuthStatusBanner
        status="unknown"
        isChecking
        credentialsConfigured
        providerName="Claude Code"
        testIdPrefix={PREFIX}
      />,
    );
    expect(screen.getByTestId(`${PREFIX}-checking`)).toBeInTheDocument();
    expect(
      screen.queryByTestId(`${PREFIX}-configured`),
    ).not.toBeInTheDocument();
  });
});
