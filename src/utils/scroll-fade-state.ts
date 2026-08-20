const SCROLL_EDGE_THRESHOLD_PX = 1;

export interface ScrollFadeState {
  left: boolean;
  right: boolean;
}

/** Whether a horizontal scroller is clipped on each edge. */
export function readScrollFadeState(element: HTMLElement): ScrollFadeState {
  const { scrollLeft, scrollWidth, clientWidth } = element;
  const maxScroll = scrollWidth - clientWidth;
  const hasOverflow = maxScroll > SCROLL_EDGE_THRESHOLD_PX;

  return {
    left: hasOverflow && scrollLeft > SCROLL_EDGE_THRESHOLD_PX,
    right: hasOverflow && scrollLeft < maxScroll - SCROLL_EDGE_THRESHOLD_PX,
  };
}
