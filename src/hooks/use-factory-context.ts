import { useNavigation } from "#/context/navigation-context";
import { FACTORIES_PATH, FACTORY_CONTEXT_PARAM } from "#/constants/factories";

const FACTORY_PAGE = new RegExp(`^${FACTORIES_PATH}/([^/]+)`);

/**
 * The factory the current page belongs to: a factory's own pages, or an
 * automation page that was opened from one (`?factory=<id>`).
 */
export function useFactoryContext(): string | null {
  const { currentPath, currentSearch } = useNavigation();
  const onFactoryPage = FACTORY_PAGE.exec(currentPath)?.[1];
  if (onFactoryPage) return decodeURIComponent(onFactoryPage);
  return new URLSearchParams(currentSearch).get(FACTORY_CONTEXT_PARAM) || null;
}
