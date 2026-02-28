import { useLocation } from "react-router";

const INTERMEDIATE_PAGE_PATHS = ["/accept-tos", "/onboarding"];

/**
 * Checks if the current page is an intermediate page.
 *
 * This hook is reusable for all intermediate pages. To add a new intermediate page,
 * add its path to INTERMEDIATE_PAGE_PATHS array.
 */
export const useIsOnIntermediatePage = (): boolean => {
  const { pathname } = useLocation();

  return INTERMEDIATE_PAGE_PATHS.includes(
    pathname as (typeof INTERMEDIATE_PAGE_PATHS)[number],
  );
};
