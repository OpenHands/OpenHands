import type { SuperAdminSetupStepId } from "../super-admin-setup-guide.constants";

export type GuidedTourSide = "top" | "right" | "bottom" | "left";

export interface GuidedTourStop {
  id: string;
  /**
   * The Canvas route the stop is on. Left out for a stop the admin reaches
   * by the previous stop's click, such as a setup page for the chosen entry.
   */
  route?: string;
  /** CSS selector of the element the stop points at. */
  anchor: string;
  title: string;
  body: string;
  side?: GuidedTourSide;
  /**
   * Hide Next; move on when the admin clicks an element matching this
   * selector inside the anchor, such as one template card in a list.
   */
  advanceOnClick?: string;
  /** Hide Next; end the stop when this setup step reports success. */
  waitForComplete?: SuperAdminSetupStepId;
  /**
   * Keep the whole page usable: no dimmed overlay, and controls outside the
   * anchor (such as a dropdown's option list) still take clicks.
   */
  interactive?: boolean;
  /**
   * Hide the popover's footer (Back and the step count), such as on a first
   * stop that only waits for the admin's click.
   */
  hideFooter?: boolean;
}

export interface GuidedTour {
  id: string;
  stops: GuidedTourStop[];
}

/** Translated button labels for the tour popover. */
export interface GuidedTourLabels {
  next: string;
  previous: string;
  done: string;
}
