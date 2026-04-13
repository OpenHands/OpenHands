import { useLocation } from "react-router";
import { useState, useEffect } from "react";

const INTERMEDIATE_PAGE_PATHS = [
  "/accept-tos",
  "/onboarding",
  "/information-request",
];

/**
 * Checks if the current page is an intermediate page.
 *
 * This hook is reusable for all intermediate pages. To add a new intermediate page,
 * add its path to INTERMEDIATE_PAGE_PATHS array.
 *
 * Returns false if called outside of Router context (e.g., during hydration).
 */
export const useIsOnIntermediatePage = (): boolean => {
  const [isOnIntermediatePage, setIsOnIntermediatePage] = useState(false);
  const { pathname } = useLocation();

  useEffect(() => {
    setIsOnIntermediatePage(
      INTERMEDIATE_PAGE_PATHS.includes(
        pathname as (typeof INTERMEDIATE_PAGE_PATHS)[number],
      ),
    );
  }, [pathname]);

  return isOnIntermediatePage;
};
