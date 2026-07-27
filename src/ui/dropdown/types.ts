import type { ReactNode } from "react";

export interface DropdownOption {
  value: string;
  label: string;
  /**
   * Optional content rendered before the label in both the trigger
   * (when this option is selected) and each menu row. Used for things
   * like status indicators; not searchable.
   */
  prefix?: ReactNode;
}
