import { useQuery } from "@tanstack/react-query";
import type { PromptEnhancementAvailability } from "@openhands/typescript-client/clients";
import PromptEnhancementService from "#/api/prompt-enhancement-service";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useAcpModelContext } from "#/hooks/use-acp-model-context";
import { useChatInputLlmProfileState } from "#/hooks/use-chat-input-llm-profile-state";
import { PROMPT_ENHANCEMENT_QUERY_KEYS } from "./query-keys";

/** Why the Enhance Prompt action is unavailable. */
export type PromptEnhancementUnavailableReason =
  /** Cloud backends have no Agent Server before a conversation starts. */
  | "cloud_backend"
  /** ACP agents use their own model, not a saved LLM profile. */
  | "acp_agent"
  | "no_profile"
  | "checking"
  /** The Agent Server predates `prompt_enhancement_v1`. */
  | "unsupported_backend"
  | "profile_unavailable"
  | "unsupported_configuration"
  | "check_failed";

export type PromptEnhancementAvailabilityState =
  | { isAvailable: true; profileName: string }
  | { isAvailable: false; reason: PromptEnhancementUnavailableReason };

const toUnavailableReason = (
  availability: Extract<PromptEnhancementAvailability, { available: false }>,
): PromptEnhancementUnavailableReason => {
  switch (availability.code) {
    case "unsupported_backend":
      return "unsupported_backend";
    case "profile_not_found":
    case "profile_unavailable":
      return "profile_unavailable";
    case "unsupported_configuration":
      return "unsupported_configuration";
    default:
      return "check_failed";
  }
};

/**
 * Whether the composer can enhance a draft with the LLM profile it would
 * send with. Only a local Agent Server can do it before a conversation exists;
 * the server check is cached per backend and profile.
 */
export function usePromptEnhancementAvailability(): PromptEnhancementAvailabilityState {
  const { backend } = useActiveBackend();
  const { isAcpContext } = useAcpModelContext();
  const { currentProfileName } = useChatInputLlmProfileState();
  const canCheck =
    backend.kind === "local" && !isAcpContext && !!currentProfileName;

  const { data, isError } = useQuery({
    queryKey: PROMPT_ENHANCEMENT_QUERY_KEYS.availability(
      backend.id,
      backend.host,
      currentProfileName,
    ),
    queryFn: () =>
      PromptEnhancementService.checkAvailability(currentProfileName ?? ""),
    enabled: canCheck,
    staleTime: 1000 * 60 * 5,
    retry: false,
  });

  if (backend.kind !== "local") {
    return { isAvailable: false, reason: "cloud_backend" };
  }
  if (isAcpContext) return { isAvailable: false, reason: "acp_agent" };
  if (!currentProfileName) return { isAvailable: false, reason: "no_profile" };
  if (isError) return { isAvailable: false, reason: "check_failed" };
  if (!data) return { isAvailable: false, reason: "checking" };
  if (!data.available) {
    return { isAvailable: false, reason: toUnavailableReason(data) };
  }
  return { isAvailable: true, profileName: currentProfileName };
}
