import { useEffect, useState, RefObject } from "react";

interface ElementDimensions {
  width: number;
  height: number;
  offsetParent: HTMLElement | null;
  computedStyle: {
    display: string;
    position: string;
    visibility: string;
  };
}

/**
 * Hook to monitor element dimensions and computed styles in real-time
 * Useful for debugging layout issues
 */
export const useElementDimensions = (
  ref: RefObject<HTMLElement | null>,
  enabled: boolean = true,
): ElementDimensions => {
  const [dimensions, setDimensions] = useState<ElementDimensions>({
    width: 0,
    height: 0,
    offsetParent: null,
    computedStyle: {
      display: "",
      position: "",
      visibility: "",
    },
  });

  useEffect(() => {
    if (!ref.current || !enabled) {
      return undefined;
    }

    const updateDimensions = () => {
      if (ref.current) {
        const computed = window.getComputedStyle(ref.current);
        const newDimensions = {
          width: ref.current.clientWidth,
          height: ref.current.clientHeight,
          offsetParent: ref.current.offsetParent as HTMLElement | null,
          computedStyle: {
            display: computed.display,
            position: computed.position,
            visibility: computed.visibility,
          },
        };

        setDimensions(newDimensions);
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(ref.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [ref, enabled]);

  return dimensions;
};
