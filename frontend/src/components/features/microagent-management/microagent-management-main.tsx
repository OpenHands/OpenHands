import { useMicroagentManagementStore } from "#/stores/microagent-management-store";
import { MicroagentManagementDefault } from "./microagent-management-default";
import { MicroagentManagementOpeningPr } from "./microagent-management-opening-pr";
import { MicroagentManagementReviewPr } from "./microagent-management-review-pr";
import { MicroagentManagementViewMicroagent } from "./microagent-management-view-microagent";
import { MicroagentManagementError } from "./microagent-management-error";
import { MicroagentManagementConversationStopped } from "./microagent-management-conversation-stopped";

export function MicroagentManagementMain() {
  const { selectedMicroagentItem } = useMicroagentManagementStore();

  const { microagent, conversation } = selectedMicroagentItem ?? {};

  if (microagent) {
    return <MicroagentManagementViewMicroagent />;
  }

  if (conversation) {
    if (conversation.pr_number && conversation.pr_number.length > 0) {
      return <MicroagentManagementReviewPr />;
    }

    const isConversationStarting =
      conversation.status === "STARTING" ||
      conversation.runtime_status === "STATUS$STARTING_RUNTIME";
    const isConversationOpeningPr =
      conversation.status === "RUNNING" &&
      conversation.runtime_status === "STATUS$READY";

    if (isConversationStarting || isConversationOpeningPr) {
      return <MicroagentManagementOpeningPr />;
    }

    if (conversation.runtime_status === "STATUS$ERROR") {
      return <MicroagentManagementError />;
    }

    if (
      conversation.status === "STOPPED" ||
      conversation.runtime_status === "STATUS$STOPPED"
    ) {
      return <MicroagentManagementConversationStopped />;
    }

    return <MicroagentManagementDefault />;
  }

  return <MicroagentManagementDefault />;
}
