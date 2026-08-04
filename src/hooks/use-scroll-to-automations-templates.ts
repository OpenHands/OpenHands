import { useEffect } from "react";
import { useLocation } from "react-router";
import {
  AUTOMATIONS_TEMPLATES_HASH,
  AUTOMATIONS_TEMPLATES_SECTION_ID,
} from "#/components/features/automations/automations-page.constants";

interface UseScrollToAutomationsTemplatesOptions {
  isReady: boolean;
}

/** Scrolls the automations catalog into view when the URL hash targets templates. */
export function useScrollToAutomationsTemplates({
  isReady,
}: UseScrollToAutomationsTemplatesOptions) {
  const { hash } = useLocation();

  useEffect(() => {
    if (!isReady || hash !== `#${AUTOMATIONS_TEMPLATES_HASH}`) {
      return undefined;
    }

    const scrollToTemplates = () => {
      document
        .getElementById(AUTOMATIONS_TEMPLATES_SECTION_ID)
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    };

    const frameId = window.requestAnimationFrame(scrollToTemplates);
    return () => window.cancelAnimationFrame(frameId);
  }, [hash, isReady]);
}
