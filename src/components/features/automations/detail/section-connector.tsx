import { ArrowDown } from "lucide-react";

/** The arrow from what starts an automation to what it does. */
export function SectionConnector() {
  return (
    <div
      aria-hidden="true"
      data-testid="automation-section-connector"
      className="flex justify-center py-1.5 text-muted"
    >
      <ArrowDown className="size-4" />
    </div>
  );
}
