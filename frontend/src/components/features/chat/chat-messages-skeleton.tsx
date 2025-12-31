import React from "react";

function MessageSkeleton({ type }: { type: "user" | "agent" }) {
  return (
    <div
      className={`rounded-xl flex flex-col gap-2 w-[60%] max-w-[400px] ${
        type === "user"
          ? "p-4 bg-tertiary self-end"
          : "mt-6 w-full bg-transparent"
      }`}
    >
      <div className="h-3 w-full skeleton !rounded-sm" />
      <div className="h-3 w-[70%] skeleton !rounded-sm" />
    </div>
  );
}

function EventSkeleton() {
  return (
    <div className="flex flex-col gap-2 border-l-2 pl-2 my-2 py-2 border-neutral-300 w-full">
      <div className="h-3 w-[40%] skeleton !rounded-sm" />
    </div>
  );
}

export function ChatMessagesSkeleton() {
  return (
    <div
      className="flex flex-col gap-4 w-full"
      data-testid="chat-messages-skeleton"
    >
      <MessageSkeleton type="user" />
      <MessageSkeleton type="agent" />
      <EventSkeleton />
      <EventSkeleton />
    </div>
  );
}
