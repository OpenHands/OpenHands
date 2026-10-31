import {
  PromptEnhancementClient,
  type PromptEnhancementAvailability,
} from "@openhands/typescript-client/clients";
import { getAgentServerClientOptions } from "./agent-server-client-options";

const createClient = () => {
  const { host, apiKey } = getAgentServerClientOptions();
  return new PromptEnhancementClient({ host, apiKey });
};

/**
 * Standalone draft enhancement on the active local Agent Server
 * (`/api/prompt-enhancement/*`). The server resolves the saved profile's
 * credentials and makes one LLM call without a conversation, events, or tools.
 * Draft text is never logged here.
 */
const PromptEnhancementService = {
  /**
   * Whether the server supports the operation and can resolve the profile.
   * Does not contact the model provider.
   */
  checkAvailability(
    profileName: string,
  ): Promise<PromptEnhancementAvailability> {
    return createClient().checkAvailability(profileName);
  },

  /** Returns the enhanced text. Rejects on abort, timeout, or a server error. */
  async enhancePrompt(
    profileName: string,
    text: string,
    signal: AbortSignal,
  ): Promise<string> {
    const response = await createClient().enhancePrompt(
      { profile_name: profileName, text },
      { signal },
    );
    return response.enhanced_text;
  },
};

export default PromptEnhancementService;
