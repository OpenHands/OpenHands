import { Tooltip } from "@heroui/react";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import { mapProvider } from "#/utils/map-provider";
import { cn } from "#/utils/utils";

interface ConversationModelBadgeProps {
  llmModel?: string | null;
}

export function ConversationModelBadge({
  llmModel,
}: ConversationModelBadgeProps) {
  if (!llmModel) return null;

  const { provider, model } = extractModelAndProvider(llmModel);
  const displayProvider = provider ? mapProvider(provider) : "";
  const displayText = provider ? `${displayProvider}/${model}` : model;

  return (
    <Tooltip content={llmModel} placement="top">
      <span
        className={cn(
          "inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 cursor-help bg-neutral-600/30 text-neutral-300",
        )}
      >
        {displayText}
      </span>
    </Tooltip>
  );
}
