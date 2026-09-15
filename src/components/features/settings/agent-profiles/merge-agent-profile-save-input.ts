import type {
  AgentProfile,
  AgentProfileSaveInput,
  OpenHandsAgentProfile,
} from "@openhands/typescript-client";

type OpenHandsAgentProfileSaveInput = Extract<
  AgentProfileSaveInput,
  { agent_kind: "openhands" }
>;

type PartialVerification = Partial<OpenHandsAgentProfile["verification"]>;

type EditedAgentProfileSaveInput =
  | (Omit<OpenHandsAgentProfileSaveInput, "verification"> & {
      verification?: PartialVerification;
    })
  | (Omit<
      Extract<AgentProfileSaveInput, { agent_kind: "acp" }>,
      "acp_server"
    > & {
      acp_server?: string;
    });

export function mergeAgentProfileSaveInput(
  stored: AgentProfile | null,
  edited: EditedAgentProfileSaveInput,
): AgentProfileSaveInput {
  if (!stored) {
    return edited as AgentProfileSaveInput;
  }

  if (stored.agent_kind !== edited.agent_kind) {
    return edited as AgentProfileSaveInput;
  }

  const { id, name, revision, ...preserved } = stored;

  if (stored.agent_kind === "openhands" && edited.agent_kind === "openhands") {
    const { verification: editedVerification, ...editedWithoutVerification } =
      edited;

    const mergedVerification: OpenHandsAgentProfile["verification"] = {
      ...stored.verification,
      ...(editedVerification ?? {}),
    };

    return {
      ...preserved,
      ...editedWithoutVerification,
      verification: mergedVerification,
    } as AgentProfileSaveInput;
  }

  return {
    ...preserved,
    ...edited,
  } as AgentProfileSaveInput;
}
