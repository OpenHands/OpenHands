import { Provider } from "#/types/settings";

export const convertRawProvidersToList = (
  raw: Partial<Record<Provider, string | null>> | undefined,
): Provider[] => {
  if (!raw) return [];
  const list: Provider[] = [];
  for (const key of Object.keys(raw)) {
    if (key) {
      list.push(key as Provider);
    }
  }
  return list;
};
