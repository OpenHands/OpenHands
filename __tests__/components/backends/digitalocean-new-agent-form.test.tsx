import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";

import type { MarsSandboxSize } from "#/api/mars/mars-tunnel-backend";
import { DigitalOceanNewAgentForm } from "#/components/features/backends/managed-agents/digitalocean-new-agent-form";

// Real English strings with interpolation, so option labels can be asserted.
vi.mock("react-i18next", async (importOriginal) => {
  const { default: translations } = await import("#/i18n/translation.json");
  const english = translations as Record<string, { en: string }>;
  return {
    ...(await importOriginal<typeof import("react-i18next")>()),
    useTranslation: () => ({
      t: (key: string, params: Record<string, unknown> = {}) =>
        (english[key]?.en ?? key).replace(/{{(\w+)}}/g, (_, name: string) =>
          String(params[name]),
        ),
      i18n: { language: "en" },
    }),
  };
});

const SIZES: MarsSandboxSize[] = [
  { slug: "mars-1vcpu-1gb", vcpus: 1, memory_mb: 1024 },
  {
    slug: "mars-2vcpu-4gb",
    vcpus: 2,
    memory_mb: 4096,
    price_per_hour_usd: 0.036,
    price_per_month_usd: 24,
  },
  {
    slug: "mars-16vcpu-32gb",
    vcpus: 16,
    memory_mb: 32768,
    price_per_hour_usd: 0.286,
    price_per_month_usd: 192.5,
  },
];

function installBridge(listSandboxSizes: () => Promise<MarsSandboxSize[]>) {
  const bridge = { listSandboxSizes: vi.fn(listSandboxSizes) };
  window.marsBridge = bridge as unknown as NonNullable<
    typeof window.marsBridge
  >;
  return bridge;
}

function renderForm() {
  const onSubmit = vi.fn();
  renderWithProviders(
    <DigitalOceanNewAgentForm
      isCreating={false}
      error={null}
      onSubmit={onSubmit}
      onCancel={vi.fn()}
    />,
  );
  return { onSubmit };
}

async function typeName(user: ReturnType<typeof userEvent.setup>) {
  const nameInput = screen.getByTestId("digitalocean-new-agent-name");
  await user.clear(nameInput);
  await user.type(nameInput, "team-agent");
}

afterEach(() => {
  delete window.marsBridge;
});

describe("DigitalOceanNewAgentForm – sandbox size", () => {
  it("lists DigitalOcean's sizes and submits the chosen slug", async () => {
    // Arrange
    installBridge(async () => SIZES);
    const user = userEvent.setup();
    const { onSubmit } = renderForm();
    await typeName(user);

    // Act
    await user.click(screen.getByRole("combobox"));
    expect(
      await screen.findByText("2 vCPU · 4 GB · $24.00/month"),
    ).toBeInTheDocument();
    await user.click(
      await screen.findByText("16 vCPU · 32 GB · $192.50/month"),
    );
    await user.click(screen.getByTestId("digitalocean-new-agent-submit"));

    // Assert
    expect(onSubmit).toHaveBeenCalledWith({
      name: "team-agent",
      size: "mars-16vcpu-32gb",
    });
  });

  it("omits the price when DigitalOcean does not quote one", async () => {
    installBridge(async () => [
      { slug: "mars-4vcpu-8gb", vcpus: 4, memory_mb: 8192 },
    ]);
    const user = userEvent.setup();
    renderForm();

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByText("4 vCPU · 8 GB")).toBeInTheDocument();
  });

  it("selects and submits the smallest size without user intervention", async () => {
    // Arrange: the catalog need not arrive smallest first.
    installBridge(async () => [...SIZES].reverse());
    const user = userEvent.setup();
    const { onSubmit } = renderForm();
    await typeName(user);
    await waitFor(() =>
      expect(screen.getByRole("combobox")).toHaveValue("1 vCPU · 1 GB"),
    );

    // Act
    await user.click(screen.getByTestId("digitalocean-new-agent-submit"));

    // Assert
    expect(onSubmit).toHaveBeenCalledWith({
      name: "team-agent",
      size: "mars-1vcpu-1gb",
    });
  });

  it("keeps DigitalOcean default as an explicit choice", async () => {
    installBridge(async () => SIZES);
    const user = userEvent.setup();
    const { onSubmit } = renderForm();
    await typeName(user);
    await waitFor(() =>
      expect(screen.getByRole("combobox")).toHaveValue("1 vCPU · 1 GB"),
    );

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Default (DigitalOcean picks)"));
    await user.click(screen.getByTestId("digitalocean-new-agent-submit"));

    expect(onSubmit).toHaveBeenCalledWith({
      name: "team-agent",
      size: undefined,
    });
  });

  it("preserves an explicit provider default when a delayed catalog arrives", async () => {
    let resolveSizes!: (sizes: MarsSandboxSize[]) => void;
    installBridge(
      () =>
        new Promise((resolve) => {
          resolveSizes = resolve;
        }),
    );
    const user = userEvent.setup();
    const { onSubmit } = renderForm();
    await typeName(user);
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Default (DigitalOcean picks)"));

    resolveSizes(SIZES);
    await user.click(screen.getByRole("combobox"));
    expect(
      await screen.findByText("16 vCPU · 32 GB · $192.50/month"),
    ).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(screen.getByRole("combobox")).toHaveValue(
      "Default (DigitalOcean picks)",
    );
    await user.click(screen.getByTestId("digitalocean-new-agent-submit"));
    expect(onSubmit).toHaveBeenCalledWith({
      name: "team-agent",
      size: undefined,
    });
  });

  it("waits for a delayed catalog instead of submitting the provider default", async () => {
    let resolveSizes!: (sizes: MarsSandboxSize[]) => void;
    installBridge(
      () =>
        new Promise((resolve) => {
          resolveSizes = resolve;
        }),
    );
    const user = userEvent.setup();
    const { onSubmit } = renderForm();
    await typeName(user);

    expect(screen.getByTestId("digitalocean-new-agent-submit")).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
    resolveSizes(SIZES);
    await waitFor(() =>
      expect(screen.getByRole("combobox")).toHaveValue("1 vCPU · 1 GB"),
    );
    await user.click(screen.getByTestId("digitalocean-new-agent-submit"));
    expect(onSubmit).toHaveBeenCalledWith({
      name: "team-agent",
      size: "mars-1vcpu-1gb",
    });
  });

  it.each(["failed", "empty"])(
    "requires an explicit default choice for a %s catalog",
    async (catalog) => {
      installBridge(async () => {
        if (catalog === "failed") throw new Error("403 forbidden");
        return [];
      });
      const user = userEvent.setup();
      const { onSubmit } = renderForm();
      await typeName(user);

      expect(
        screen.getByTestId("digitalocean-new-agent-submit"),
      ).toBeDisabled();
      await user.click(screen.getByRole("combobox"));
      await user.click(await screen.findByText("Default (DigitalOcean picks)"));
      await user.click(screen.getByTestId("digitalocean-new-agent-submit"));

      expect(onSubmit).toHaveBeenCalledWith({
        name: "team-agent",
        size: undefined,
      });
    },
  );
});
