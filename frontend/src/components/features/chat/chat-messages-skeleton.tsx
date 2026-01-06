import React from "react";

// Fixed pattern to prevent layout shifts during initial message loading
const SKELETON_PATTERN = [
  { width: "w-[25%]", height: "h-4", align: "justify-end" }, // User
  { width: "w-[60%]", height: "h-4", align: "justify-start" }, // Agent
  { width: "w-[45%]", height: "h-4", align: "justify-start" }, // Agent
  { width: "w-[85%]", height: "h-20", align: "justify-start" }, // Large block for Action/Terminal outputs
  { width: "w-[35%]", height: "h-4", align: "justify-end" }, // User
  { width: "w-[50%]", height: "h-4", align: "justify-start" }, // Agent
];

function SkeletonBlock({ width, height }: { width: string; height: string }) {
  // Using low opacity foreground for a subtle fading effect during transitions
  return (
    <div
      className={`rounded-md bg-foreground/5 animate-pulse ${width} ${height}`}
    />
  );
}

export function ChatMessagesSkeleton() {
  return (
    <div
      className="flex flex-col gap-6 p-4 w-full"
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
