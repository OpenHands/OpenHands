import { describe, expect, it } from "vitest";

import { formatShortDate } from "#/components/features/admin-dashboard/usage-dashboard-utils";

describe("formatShortDate", () => {
  // Date-only API strings parse as UTC midnight; labels must not shift
  // with the viewer's timezone (previously "2026-07-01" showed "Jun 30"
  // for viewers west of UTC).
  it("renders the UTC calendar date regardless of local timezone", () => {
    expect(formatShortDate("2026-07-01")).toBe("Jul 1");
    expect(formatShortDate("2026-06-23")).toBe("Jun 23");
    expect(formatShortDate("2026-12-31")).toBe("Dec 31");
  });

  it("renders UTC-midnight timestamps on their UTC day", () => {
    expect(formatShortDate("2026-07-01T00:00:00Z")).toBe("Jul 1");
  });
});
