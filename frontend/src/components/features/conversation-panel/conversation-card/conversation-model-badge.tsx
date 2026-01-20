import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";
import { mapProvider } from "#/utils/map-provider";
import { Typography } from "#/ui/typography";

interface ConversationModelBadgeProps {
  llmModel: string;
}

export function ConversationModelBadge({
  llmModel,
}: ConversationModelBadgeProps) {
  const { provider, model } = extractModelAndProvider(llmModel);
  const displayProvider = provider ? mapProvider(provider) : "";
  const displayText = provider ? `${displayProvider}/${model}` : model;

  return (
    <StyledTooltip content={llmModel} placement="top">
      <Typography.Text className="inline-flex cursor-pointer items-center px-1.5 py-0.5 rounded text-[10px] font-medium shrink-0 bg-neutral-600/30 text-neutral-300 truncate max-w-[75px]">
        {displayText}
      </Typography.Text>
    </StyledTooltip>
  );
}
