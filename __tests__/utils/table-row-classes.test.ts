import { describe, expect, it } from "vitest";
import {
  tableCellClassName,
  tableRowHeightClassName,
  tableRowHoverClassName,
} from "#/utils/table-row-classes";

describe("tableRowClasses", () => {
  it("uses a fixed 44px row height and subtle hover token", () => {
    expect(tableRowHeightClassName).toBe("h-11");
    expect(tableCellClassName).toContain("px-3");
    expect(tableCellClassName).toContain("align-middle");
    expect(tableRowHoverClassName).toContain("hover:bg-interactive-hover-low");
  });
});
