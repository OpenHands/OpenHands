import React from "react";
import ReactJsonView from "@microlink/react-json-view";
import { useTranslation } from "react-i18next";
import { MCPObservation } from "#/types/core/observations";
import { JSON_VIEW_THEME } from "#/utils/constants";

interface MCPObservationContentProps {
  event: MCPObservation;
}

export function MCPObservationContent({ event }: MCPObservationContentProps) {
  const { t } = useTranslation();

  // Parse the content as JSON if possible
  let outputData: unknown;
  try {
    outputData = JSON.parse(event.content);
  } catch (e) {
    // If parsing fails, use the raw content
    outputData = event.content;
  }

  const hasArguments =
    event.extras.arguments && Object.keys(event.extras.arguments).length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Arguments section */}
      {hasArguments && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-content">
              {t("MCP_OBSERVATION$ARGUMENTS")}
            </h3>
          </div>
          <div className="p-3 bg-code-surface rounded-md overflow-auto text-content max-h-[200px] shadow-inner">
            <ReactJsonView
              name={false}
              src={event.extras.arguments}
              theme={JSON_VIEW_THEME}
              collapsed={1}
              displayDataTypes={false}
            />
          </div>
        </div>
      )}

      {/* Output section */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-content">
            {t("MCP_OBSERVATION$OUTPUT")}
          </h3>
        </div>
        <div className="p-3 bg-code-surface rounded-md overflow-auto text-content max-h-[300px] shadow-inner">
          {typeof outputData === "object" && outputData !== null ? (
            <ReactJsonView
              name={false}
              src={outputData}
              theme={JSON_VIEW_THEME}
              collapsed={1}
              displayDataTypes={false}
            />
          ) : (
            <pre className="whitespace-pre-wrap">
              {event.content.trim() || t("OBSERVATION$MCP_NO_OUTPUT")}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}
