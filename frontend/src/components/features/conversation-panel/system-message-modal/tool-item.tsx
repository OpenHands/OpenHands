import { MouseEvent, useEffect, useState } from "react";
import { ToolParameters } from "./tool-parameters";
import { ChatCompletionToolParam } from "#/types/v1/core";
import { MarkdownRenderer } from "../../markdown/markdown-renderer";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";

interface FunctionData {
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

interface ToolData {
  // V0/OpenAI format
  type?: string;
  function?: FunctionData;
  name?: string;
  description?: string;
  parameters?: Record<string, unknown>;
  // V1 format
  title?: string;
  kind?: string;
  annotations?: {
    title?: string;
  };
}

interface ToolItemProps {
  tool: Record<string, unknown> | ChatCompletionToolParam;
  index: number;
  isExpanded: boolean;
  onToggle: (index: number) => void;
}

export interface ToolDisplayMetadata {
  name: string;
  description: string;
  parameters: Record<string, unknown> | null;
  kind: string | null;
}

export function getToolDisplayMetadata(
  tool: Record<string, unknown> | ChatCompletionToolParam,
): ToolDisplayMetadata {
  const toolData = tool as ToolData;
  const functionData = toolData.function || toolData;

  const name =
    toolData.title ||
    toolData.annotations?.title ||
    functionData.name ||
    (toolData.type === "function" && toolData.function?.name) ||
    "";

  const description =
    toolData.description ||
    functionData.description ||
    (toolData.type === "function" && toolData.function?.description) ||
    "";

  const parameters =
    functionData.parameters ||
    (toolData.type === "function" && toolData.function?.parameters) ||
    toolData.parameters ||
    null;

  const kind = toolData.kind || toolData.type || null;

  return {
    name: String(name),
    description: String(description),
    parameters,
    kind,
  };
}

export function ToolItem({ tool, index, isExpanded, onToggle }: ToolItemProps) {
  const [copyMode, setCopyMode] = useState<"copy" | "copied">("copy");
  const { name, description, parameters, kind } = getToolDisplayMetadata(tool);

  useEffect(() => {
    if (copyMode !== "copied") {
      return undefined;
    }

    const timeoutId = setTimeout(() => setCopyMode("copy"), 2000);
    return () => clearTimeout(timeoutId);
  }, [copyMode]);

  const onCopy = async (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    await navigator.clipboard.writeText(name);
    setCopyMode("copied");
  };

  return (
    <div className="rounded-md overflow-hidden">
      <div className="flex items-stretch gap-2">
        <button
          type="button"
          data-testid="toggle-button"
          onClick={() => onToggle(index)}
          className="w-full py-3 px-2 text-left flex items-center justify-between hover:bg-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            <span className="font-bold text-gray-100">{name}</span>
            {kind && (
              <span className="px-2 py-1 text-xs rounded-full bg-gray-800 text-gray-200">
                {kind}
              </span>
            )}
          </div>
          <span className="text-gray-300">{isExpanded ? "v" : ">"}</span>
        </button>
        <div className="flex items-center">
          <CopyToClipboardButton
            isHidden={false}
            isDisabled={copyMode === "copied"}
            onClick={onCopy}
            mode={copyMode}
          />
        </div>
      </div>

      {isExpanded && (
        <div className="px-2 pb-3 pt-1">
          <div className="mt-2 mb-3 text-sm text-gray-300 leading-relaxed">
            <MarkdownRenderer>{description}</MarkdownRenderer>
          </div>

          {/* Parameters section */}
          {parameters && <ToolParameters parameters={parameters} />}
        </div>
      )}
    </div>
  );
}
