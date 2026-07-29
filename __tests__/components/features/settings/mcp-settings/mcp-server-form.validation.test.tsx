import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MCPServerForm } from "#/components/features/settings/mcp-settings/mcp-server-form";

// i18n mock
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("MCPServerForm validation", () => {
  const noop = () => {};

  it("rejects invalid env var lines and allows blank lines", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "stdio" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    // Fill required fields
    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "my_server" },
    });
    fireEvent.change(screen.getByTestId("command-input"), {
      target: { value: "npx" },
    });

    // Invalid env entries mixed with blank lines
    fireEvent.change(screen.getByTestId("env-input"), {
      target: { value: "invalid\n\nKEY=value\n=novalue\nKEY_ONLY=" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    // Should show invalid env format error
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_ENV_INVALID_FORMAT"),
    ).toBeInTheDocument();

    // Fix env with valid lines and blank lines
    fireEvent.change(screen.getByTestId("env-input"), {
      target: { value: "KEY=value\n\nANOTHER=123" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    // No error; submit should be called
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("includes an optional server name for sse/shttp servers", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "sse" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("server-name-input"), {
      target: { value: "my_search" },
    });
    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      type: "sse",
      name: "my_search",
      url: "https://api.example.com",
    });
  });

  it("allows hyphenated local stdio server names", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "stdio" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "integrations-hub" },
    });
    fireEvent.change(screen.getByTestId("command-input"), {
      target: { value: "npx" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      type: "stdio",
      name: "integrations-hub",
      command: "npx",
    });
  });

  it("allows hyphenated remote server names", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "shttp" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("server-name-input"), {
      target: { value: "integrations-hub" },
    });
    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://api.example.com/mcp" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      type: "shttp",
      name: "integrations-hub",
      url: "https://api.example.com/mcp",
    });
  });

  it("submits header authentication as a tagged auth credential", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="edit"
        server={{
          id: "shttp-0",
          type: "shttp",
          name: "datadog",
          url: "https://api.example.com/mcp",
          auth: {
            strategy: "header",
            headers: {
              "DD-API-KEY": "",
              "DD-APPLICATION-KEY": "",
            },
          },
        }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("headers-input"), {
      target: {
        value: "DD-API-KEY=dd-api\nDD-APPLICATION-KEY=dd-app",
      },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      type: "shttp",
      name: "datadog",
      url: "https://api.example.com/mcp",
      auth: {
        strategy: "header",
        headers: {
          "DD-API-KEY": "dd-api",
          "DD-APPLICATION-KEY": "dd-app",
        },
      },
    });
  });

  it("rejects an sse/shttp server name with unsafe characters", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "sse" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("server-name-input"), {
      target: { value: "my server" },
    });
    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    // A name with a space can't be a safe mcp_config key, so submission is
    // blocked rather than persisted under a malformed key.
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_NAME_INVALID"),
    ).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("omits the name when the sse/shttp name field is left blank", () => {
    const onSubmit = vi.fn();

    render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "shttp" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0].name).toBeUndefined();
  });

  it("preserves OAuth state when editing a remote OAuth server", () => {
    const onSubmit = vi.fn();
    const oauthState = {
      tokens: {
        access_token: "**********",
        refresh_token: "**********",
      },
    };

    render(
      <MCPServerForm
        mode="edit"
        server={{
          id: "shttp-0",
          type: "shttp",
          name: "superhuman_mail",
          url: "https://mcp.mail.superhuman.com/mcp",
          auth: {
            strategy: "oauth2",
            authentication: { type: "oauth", client_auth_method: "none" },
            state: oauthState,
          },
        }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.click(screen.getByTestId("submit-button"));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_auth_method: "none" },
        state: oauthState,
      },
    });
  });

  it("rejects duplicate URLs across sse/shttp types", () => {
    const onSubmit = vi.fn();

    const existingServers = [
      { id: "sse-1", type: "sse" as const, url: "https://api.example.com" },
      { id: "shttp-1", type: "shttp" as const, url: "https://x.example.com" },
    ];

    const r1 = render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "sse" }}
        existingServers={existingServers}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getAllByTestId("url-input")[0], {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getAllByTestId("submit-button")[0]);
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_URL_DUPLICATE"),
    ).toBeInTheDocument();

    // Unmount first form, then check shttp duplicate
    r1.unmount();

    const r2 = render(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp2", type: "shttp" }}
        existingServers={existingServers}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getAllByTestId("url-input")[0], {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getAllByTestId("submit-button")[0]);
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_URL_DUPLICATE"),
    ).toBeInTheDocument();

    r2.unmount();
  });
});
