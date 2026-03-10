import { useState, useEffect, useRef } from "react";

const MOBILE_BREAKPOINT = 1024;

/**
 * Returns true when window width is at or below the breakpoint.
 * Only triggers a re-render when the boolean value changes (i.e., when the
 * width crosses the breakpoint), NOT on every pixel of resize.
 *
 * This replaces useWindowSize() for breakpoint detection, avoiding
 * unnecessary re-renders during drag resize.
 */
export function useBreakpoint(breakpoint: number = MOBILE_BREAKPOINT): boolean {
  const [isMobile, setIsMobile] = useState(
    () => window.innerWidth <= breakpoint,
  );
  const isMobileRef = useRef(isMobile);

  useEffect(() => {
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
