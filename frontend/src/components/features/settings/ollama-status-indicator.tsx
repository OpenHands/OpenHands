import { useOllamaModels } from "#/hooks/query/use-ollama-models";
import { cn } from "#/utils/utils";

interface OllamaStatusIndicatorProps {
  baseUrl: string | null;
}

export function OllamaStatusIndicator({ baseUrl }: OllamaStatusIndicatorProps) {
  const { data, isLoading, isError } = useOllamaModels(baseUrl);

  const isConnected = data && data.length > 0;
  const isEmpty = !isLoading && !isError && data && data.length === 0;

  if (!baseUrl) return null;

  return (
    <div className="flex items-center gap-2 text-xs">
      <span
        className={cn(
          "w-2 h-2 rounded-full",
          isLoading && "bg-yellow-500",
          isConnected && "bg-green-500",
          (isError || isEmpty) && "bg-red-500",
        )}
      />
      <span className="text-tertiary-alt">
        {isLoading && "Connecting..."}
        {isConnected &&
          `${data.length} model${data.length === 1 ? "" : "s"} found`}
        {isError && "Cannot connect to Ollama"}
        {isEmpty && "No models found"}
      </span>
    </div>
  );
}
