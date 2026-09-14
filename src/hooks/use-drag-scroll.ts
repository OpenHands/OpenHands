import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";

/** Movement required before a mouse gesture is treated as a scroll drag. */
const DRAG_COMMIT_PX = 2;

/**
 * Mouse drag-to-scroll for a horizontal `overflow-x-auto` container. Touch
 * input keeps the browser's native panning; only primary-button mouse drags
 * are handled. After a committed drag, the click that follows mouseup is
 * suppressed so child buttons are not activated by the drag release.
 */
export function useDragScroll(scrollRef: RefObject<HTMLElement | null>) {
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);
  const startScrollLeftRef = useRef(0);
  const dragCommittedRef = useRef(false);

  const handleMouseDown = useCallback(
    (event: React.MouseEvent) => {
      const element = scrollRef.current;
      if (event.button !== 0 || !element) {
        return;
      }
      dragCommittedRef.current = false;
      startXRef.current = event.clientX;
      startScrollLeftRef.current = element.scrollLeft;
      setIsDragging(true);
    },
    [scrollRef],
  );

  const handleClickCapture = useCallback((event: React.MouseEvent) => {
    if (!dragCommittedRef.current) {
      return;
    }
    dragCommittedRef.current = false;
    event.preventDefault();
    event.stopPropagation();
  }, []);

  const handleDragStart = useCallback((event: React.DragEvent) => {
    // Child <img> elements would otherwise start a native image drag.
    event.preventDefault();
  }, []);

  useEffect(() => {
    if (!isDragging) {
      return undefined;
    }

    const handleMouseMove = (event: MouseEvent) => {
      const element = scrollRef.current;
      if (!element) {
        return;
      }

      // The mouseup can be missed when released outside the window; end the
      // drag once the primary button is no longer held.
      if ((event.buttons & 1) === 0) {
        setIsDragging(false);
        return;
      }

      if (!dragCommittedRef.current) {
        if (Math.abs(event.clientX - startXRef.current) < DRAG_COMMIT_PX) {
          return;
        }
        dragCommittedRef.current = true;
      }

      event.preventDefault();
      element.scrollLeft =
        startScrollLeftRef.current - (event.clientX - startXRef.current);
    };

    const handleMouseUp = () => setIsDragging(false);

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, scrollRef]);

  return { handleMouseDown, handleClickCapture, handleDragStart };
}
