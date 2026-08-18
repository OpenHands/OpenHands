import { gunzipSync } from "node:zlib";
import { describe, expect, it, vi } from "vitest";

const BUNDLE_FILES: Record<string, Record<string, string>> = {
  "widget-monitor": { "main.py": "print('watching')\n" },
};

vi.mock("@openhands/extensions/automations", () => ({
  AUTOMATION_CATALOG: [],
  getAutomationBundleFiles: (id: string) => BUNDLE_FILES[id],
}));

const { packBundle, getBundleFiles } =
  await import("#/manifests/manifest-bundle");
const { createSetupEntry, createSetup } = await import("./manifest-test-data");

const decoder = new TextDecoder();

/** Every member of a packed bundle, keyed by name. */
function readArchive(archive: Uint8Array): Record<string, string> {
  const tar = new Uint8Array(gunzipSync(archive));
  const contents: Record<string, string> = {};
  const field = (block: Uint8Array, offset: number, size: number) =>
    decoder.decode(block.subarray(offset, offset + size)).replace(/\0.*$/, "");

  let offset = 0;
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;
    const size = parseInt(field(header, 124, 12).trim() || "0", 8);
    contents[field(header, 0, 100)] = decoder.decode(
      tar.subarray(offset + 512, offset + 512 + size),
    );
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  return contents;
}

function bundleEntry(overrides = {}) {
  return createSetupEntry({
    setup: createSetup({
      prompt: undefined,
      bundle: {
        version: "1.0.0",
        entrypoint: "python3 main.py",
        files: { "main.py": "skills/widget-monitor/scripts/main.py" },
        config: {
          repos: ["{{form.repository}}"],
          max_per_run: 3,
          dry_run: false,
        },
        ...overrides,
      },
    }),
  });
}

const VALUES = { repository: "OpenHands/automation", schedule: "*/15 * * * *" };

describe("packBundle", () => {
  it("packs the entry's files with the config the form rendered", async () => {
    // Act
    const archive = await packBundle(bundleEntry(), VALUES);

    // Assert
    const contents = readArchive(archive);
    expect(contents["main.py"]).toBe("print('watching')\n");
    expect(JSON.parse(contents["config.json"])).toEqual({
      repos: ["OpenHands/automation"],
      max_per_run: 3,
      dry_run: false,
    });
  });

  it("packs the same config the create request records as provenance", async () => {
    // Arrange: the tarball and the template config disagreeing would leave the
    // stored provenance describing a run that never happened.
    const entry = bundleEntry();
    const { buildCreatePayload } = await import("#/manifests/automation-setup");

    // Act
    const contents = readArchive(await packBundle(entry, VALUES));
    const payload = buildCreatePayload(entry, VALUES);

    // Assert
    expect(JSON.parse(contents["config.json"])).toEqual(
      (payload?.template as { config: unknown }).config,
    );
  });

  it("reports an entry the published package ships no files for", () => {
    // Act + Assert
    expect(() => getBundleFiles("not-published")).toThrow(
      /ships no bundle files/,
    );
  });

  it("reports a declared file the published package is missing", async () => {
    // Arrange
    const entry = bundleEntry({
      files: {
        "main.py": "skills/widget-monitor/scripts/main.py",
        "setup.sh": "automations/catalog/widget-monitor/setup.sh",
      },
    });

    // Act + Assert
    await expect(packBundle(entry, VALUES)).rejects.toThrow(
      /missing bundle files.*setup\.sh/,
    );
  });
});
