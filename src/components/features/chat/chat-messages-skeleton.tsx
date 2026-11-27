import React from "react";
import { useTranslation } from "react-i18next";
import { cn } from "#/utils/utils";
import { I18nKey } from "#/i18n/declaration";

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
      className={cn("rounded-md bg-foreground/5 animate-pulse", width, height)}
    />
  );
}

export function ChatMessagesSkeleton() {
  const { t } = useTranslation("openhands");

  return (
    <div
      className="flex flex-col gap-6 p-4 w-full h-full overflow-hidden"
      data-testid="chat-messages-skeleton"
      aria-label={t(I18nKey.CHAT_INTERFACE$LOADING_CONVERSATION)}
    >
      {SKELETON_PATTERN.map((item, i) => (
        <div key={i} className={cn("flex w-full", item.align)}>
          <SkeletonBlock width={item.width} height={item.height} />
        </div>
      ))}
    </div>
  );
}
