export const FACTORIES_PATH = "/factories";

/** The agent server's experimental feature that turns Factories on locally. */
export const FACTORIES_EXPERIMENTAL_FEATURE = "factories";

export function factoryPath(id: string): string {
  return `${FACTORIES_PATH}/${encodeURIComponent(id)}`;
}

/** Query param that keeps a factory's context on the shared automation pages. */
export const FACTORY_CONTEXT_PARAM = "factory";

/** Adds the factory context to `path`; without a factory it returns `path` as is. */
export function withFactoryContext(path: string, factoryId?: string | null) {
  if (!factoryId) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}${FACTORY_CONTEXT_PARAM}=${encodeURIComponent(factoryId)}`;
}
