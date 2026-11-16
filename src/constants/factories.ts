export const FACTORIES_PATH = "/factories";

export function factoryPath(id: string): string {
  return `${FACTORIES_PATH}/${encodeURIComponent(id)}`;
}
