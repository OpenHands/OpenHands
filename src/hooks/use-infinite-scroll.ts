import { useEffect, useRef, useCallback } from "react";

interface UseInfiniteScrollOptions {
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  threshold?: number;
}

export const useInfiniteScroll = ({
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
  threshold = 100,
}: UseInfiniteScrollOptions) => {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleScroll = useCallback(() => {
    if (!containerRef.current || isFetchingNextPage || !hasNextPage) {
      return;
    }

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - threshold;

    if (isNearBottom) {
      fetchNextPage();
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, threshold]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    container.addEventListener("scroll", handleScroll);
    return () => {
      container.removeEventListener("scroll", handleScroll);
    };
  }, [handleScroll]);

  return containerRef;
};
