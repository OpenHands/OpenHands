import { useState, useEffect, useRef } from "react";

const MOBILE_BREAKPOINT = 1024;

/**
 * Returns true when window width is at or below the breakpoint.
 * Only triggers a re-render when the boolean value changes (i.e., when the
 * width crosses the breakpoint), NOT on every pixel of resize.
 *
 * This replaces useWindowSize() for breakpoint detection, avoiding
 * unnecessary re-renders during drag resize.
 *
 * Note: Returns false (desktop) during SSR to avoid hydration mismatch,
 * then updates to the correct value after mount.
 */
export function useBreakpoint(breakpoint: number = MOBILE_BREAKPOINT): boolean {
  // Start with false (desktop) during SSR to avoid hydration mismatch
  const [isMobile, setIsMobile] = useState(false);
  const isMobileRef = useRef(false);

  useEffect(() => {
    // Update initial value after hydration
    const initialIsMobile = window.innerWidth <= breakpoint;
    isMobileRef.current = initialIsMobile;
    setIsMobile(initialIsMobile);

    function handleResize() {
      const newIsMobile = window.innerWidth <= breakpoint;
      if (newIsMobile !== isMobileRef.current) {
        isMobileRef.current = newIsMobile;
        setIsMobile(newIsMobile);
      }
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [breakpoint]);

  return isMobile;
}
