import { cn } from "#/utils/utils";
import { formatClock } from "#/utils/format-clock";
import { useElapsedSeconds } from "#/hooks/use-elapsed-seconds";

export function LiveElapsedClock({
  startTimestamp,
  align,
}: {
  startTimestamp: string;
  align: "left" | "right";
}) {
  const elapsedSeconds = useElapsedSeconds(startTimestamp);

  return (
    <div
      data-testid="live-elapsed-clock"
      className={cn(
        "absolute top-full z-10 flex w-max items-center gap-1.5 pt-0.5 text-xs leading-4 text-muted",
        align === "right" ? "right-0" : "left-0",
      )}
    >
      <time className="tabular-nums">{formatClock(elapsedSeconds)}</time>
    </div>
  );
}
