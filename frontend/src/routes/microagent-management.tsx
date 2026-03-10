import { queryClient } from "#/query-client-config";
import { WebClientConfig } from "#/api/option-service/option.types";
import OptionService from "#/api/option-service/option-service.api";
import { MicroagentManagementContent } from "#/components/features/microagent-management/microagent-management-content";
import { ConversationSubscriptionsProvider } from "#/context/conversation-subscriptions-provider";
import { V0EventHandler } from "#/wrapper/v0-event-handler";

export const clientLoader = async () => {
  let config = queryClient.getQueryData<WebClientConfig>(["web-client-config"]);
  if (!config) {
    config = await OptionService.getConfig();
    queryClient.setQueryData<WebClientConfig>(["web-client-config"], config);
  }

  return null;
};

function MicroagentManagement() {
  return (
    <ConversationSubscriptionsProvider>
      <V0EventHandler>
        <MicroagentManagementContent />
      </V0EventHandler>
    </ConversationSubscriptionsProvider>
  );
}

export default MicroagentManagement;
