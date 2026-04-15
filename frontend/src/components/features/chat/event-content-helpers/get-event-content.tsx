import { Trans } from "react-i18next";
import { OpenHandsAction } from "#/types/core/actions";
import { isOpenHandsAction, isOpenHandsObservation } from "#/types/core/guards";
import { OpenHandsObservation } from "#/types/core/observations";
import { MonoComponent } from "../mono-component";
import { PathComponent } from "../path-component";
import { getActionContent } from "./get-action-content";
import { getObservationContent } from "./get-observation-content";
import i18n from "#/i18n";

const hasPathProperty = (
  obj: Record<string, unknown>,
): obj is { path: string } => typeof obj.path === "string";

const hasCommandProperty = (
  obj: Record<string, unknown>,
): obj is { command: string } => typeof obj.command === "string";

const trimText = (text: string, maxLength: number): string => {
  if (!text) return "";
  return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
};

export const getEventContent = (
  event: OpenHandsAction | OpenHandsObservation,
) => {
  let title: React.ReactNode = "";
  let details: string = "";

  if (isOpenHandsAction(event)) {
    const actionKey = `ACTION_MESSAGE$${event.action.toUpperCase()}`;

    // If translation key exists, use Trans component
    if (i18n.exists(actionKey)) {
      title = (
        <Trans
          i18nKey={actionKey}
          values={{
            path: hasPathProperty(event.args) && event.args.path,
            command:
              hasCommandProperty(event.args) &&
              trimText(event.args.command, 80),
            mcp_tool_name: event.action === "call_tool_mcp" && event.args.name,
          }}
          components={{
            path: <PathComponent />,
            cmd: <MonoComponent />,
          }}
        />
      );
    } else {
      // For generic actions, just use the uppercase type
      title = event.action.toUpperCase();
    }
    details = getActionContent(event);
  }

  if (isOpenHandsObservation(event)) {
    const observationKey = `OBSERVATION_MESSAGE$${event.observation.toUpperCase()}`;

    // If translation key exists, use Trans component
    if (i18n.exists(observationKey)) {
      title = (
        <Trans
          i18nKey={observationKey}
          values={{
            path: hasPathProperty(event.extras) && event.extras.path,
            command:
              hasCommandProperty(event.extras) &&
              trimText(event.extras.command, 80),
            mcp_tool_name: event.observation === "mcp" && event.extras.name,
          }}
          components={{
            path: <PathComponent />,
            cmd: <MonoComponent />,
          }}
        />
      );
    } else {
      // For generic observations, just use the uppercase type
      title = event.observation.toUpperCase();
    }
    details = getObservationContent(event);
  }

  return {
    title: title ?? i18n.t("EVENT$UNKNOWN_EVENT"),
    details,
  };
};
