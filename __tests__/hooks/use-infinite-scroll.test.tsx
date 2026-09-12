import { fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useInfiniteScroll } from "#/hooks/use-infinite-scroll";

interface HarnessProps {
  fetchNextPage: () => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  threshold?: number;
}

function InfiniteScrollHarness(props: HarnessProps) {
  const ref = useInfiniteScroll(props);
  return <div ref={ref} data-testid="scroll-container" />;
}

function setScrollMetrics({
  clientHeight,
  element,
  scrollHeight,
  scrollTop,
}: {
  clientHeight: number;
  element: HTMLElement;
  scrollHeight: number;
  scrollTop: number;
}) {
  Object.defineProperties(element, {
    clientHeight: { configurable: true, value: clientHeight },
    scrollHeight: { configurable: true, value: scrollHeight },
    scrollTop: { configurable: true, value: scrollTop, writable: true },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useInfiniteScroll", () => {
  it("returns an unattached container ref when no element uses it", () => {
    const { result } = renderHook(() =>
      useInfiniteScroll({
        fetchNextPage: vi.fn(),
        hasNextPage: true,
        isFetchingNextPage: false,
      }),
    );

    expect(result.current.current).toBeNull();
  });

  it("fetches at the exact default threshold but not one pixel before it", () => {
    const fetchNextPage = vi.fn();
    render(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage
        isFetchingNextPage={false}
      />,
    );
    const element = screen.getByTestId("scroll-container");
    setScrollMetrics({
      clientHeight: 200,
      element,
      scrollHeight: 1000,
      scrollTop: 699,
    });

    fireEvent.scroll(element);
    expect(fetchNextPage).not.toHaveBeenCalled();

    element.scrollTop = 700;
    fireEvent.scroll(element);
    expect(fetchNextPage).toHaveBeenCalledOnce();
  });

  it("uses an updated custom threshold after rerender", () => {
    const fetchNextPage = vi.fn();
    const { rerender } = render(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage
        isFetchingNextPage={false}
        threshold={49}
      />,
    );
    const element = screen.getByTestId("scroll-container");
    setScrollMetrics({
      clientHeight: 200,
      element,
      scrollHeight: 1000,
      scrollTop: 750,
    });

    fireEvent.scroll(element);
    expect(fetchNextPage).not.toHaveBeenCalled();

    rerender(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage
        isFetchingNextPage={false}
        threshold={50}
      />,
    );
    fireEvent.scroll(element);
    expect(fetchNextPage).toHaveBeenCalledOnce();
  });

  it("waits for another page to exist and for the current fetch to finish", () => {
    const fetchNextPage = vi.fn();
    const { rerender } = render(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage={false}
        isFetchingNextPage={false}
      />,
    );
    const element = screen.getByTestId("scroll-container");
    setScrollMetrics({
      clientHeight: 200,
      element,
      scrollHeight: 1000,
      scrollTop: 800,
    });

    fireEvent.scroll(element);
    expect(fetchNextPage).not.toHaveBeenCalled();

    rerender(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage
        isFetchingNextPage
      />,
    );
    fireEvent.scroll(element);
    expect(fetchNextPage).not.toHaveBeenCalled();

    rerender(
      <InfiniteScrollHarness
        fetchNextPage={fetchNextPage}
        hasNextPage
        isFetchingNextPage={false}
      />,
    );
    fireEvent.scroll(element);
    expect(fetchNextPage).toHaveBeenCalledOnce();
  });

  it("uses the latest fetch callback and stops responding after unmount", () => {
    const firstFetch = vi.fn();
    const latestFetch = vi.fn();
    const { rerender, unmount } = render(
      <InfiniteScrollHarness
        fetchNextPage={firstFetch}
        hasNextPage
        isFetchingNextPage={false}
      />,
    );
    const element = screen.getByTestId("scroll-container");
    setScrollMetrics({
      clientHeight: 200,
      element,
      scrollHeight: 1000,
      scrollTop: 800,
    });

    rerender(
      <InfiniteScrollHarness
        fetchNextPage={latestFetch}
        hasNextPage
        isFetchingNextPage={false}
      />,
    );
    fireEvent.scroll(element);

    expect(firstFetch).not.toHaveBeenCalled();
    expect(latestFetch).toHaveBeenCalledOnce();

    unmount();
    fireEvent.scroll(element);
    expect(latestFetch).toHaveBeenCalledOnce();
  });
});
