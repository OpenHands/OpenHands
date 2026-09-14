import type { CanvasExtensionModule } from "#/types/canvas-extension";

export class InvalidCanvasExtensionModuleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidCanvasExtensionModuleError";
  }
}

export function assertCanvasExtensionModule(
  value: unknown,
): asserts value is CanvasExtensionModule {
  if (
    typeof value !== "object" ||
    value === null ||
    !("activate" in value) ||
    typeof value.activate !== "function"
  ) {
    throw new InvalidCanvasExtensionModuleError(
      'Canvas Extension entrypoint must export an "activate" function.',
    );
  }
}

/**
 * Import an authenticated, self-contained ESM bundle without putting the
 * Agent Server session key in a URL. The source has already been fetched by
 * Canvas through its authenticated HTTP client.
 */
export async function loadCanvasExtensionModule(
  source: string,
): Promise<CanvasExtensionModule> {
  const blob = new Blob([source], { type: "text/javascript" });
  const moduleUrl = URL.createObjectURL(blob);
  try {
    const imported: unknown = await import(/* @vite-ignore */ moduleUrl);
    assertCanvasExtensionModule(imported);
    return imported;
  } finally {
    URL.revokeObjectURL(moduleUrl);
  }
}
