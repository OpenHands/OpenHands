import React from "react";

const SKELETON_PATTERN = [
  { width: "w-[25%]", height: "h-4", align: "justify-end" },
  { width: "w-[60%]", height: "h-4", align: "justify-start" },
  { width: "w-[45%]", height: "h-4", align: "justify-start" },
  { width: "w-[65%]", height: "h-20", align: "justify-start" },
  { width: "w-[35%]", height: "h-4", align: "justify-end" },
  { width: "w-[50%]", height: "h-4", align: "justify-start" },
  { width: "w-[30%]", height: "h-4", align: "justify-end" },
  { width: "w-[75%]", height: "h-4", align: "justify-start" },
  { width: "w-[55%]", height: "h-4", align: "justify-start" },
];

function SkeletonBlock({ width, height }: { width: string; height: string }) {
  return (
    <div
      className={`rounded-md bg-foreground/5 animate-pulse ${width} ${height}`}
    />
  );
}

export function ChatMessagesSkeleton() {
  return (
    <div
      className="flex flex-col gap-6 p-4 w-full h-full overflow-hidden"
      data-testid="chat-messages-skeleton"
      aria-label="Loading conversation"
    >
      {SKELETON_PATTERN.map((item, i) => (
        <div key={i} className={`flex w-full ${item.align}`}>
          <SkeletonBlock width={item.width} height={item.height} />
        </div>
      ))}
    </div>
  );
}
