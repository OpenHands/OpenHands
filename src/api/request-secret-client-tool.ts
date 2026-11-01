import {
  REQUEST_SECRET_TOOL_NAME,
  SECRET_REQUEST_RESULT_PREFIX,
} from "#/constants/secret-request";
import type { ClientToolSpec } from "./canvas-ui-client-tool";

const REQUEST_SECRET_DESCRIPTION = `Ask the user to securely provide a secret you need (an API key, token, password or other credential) through a form shown in this conversation, without sending them to Settings.

Call this when a task needs a credential that is not among your available
secrets — for example a GitHub token to push to a repository. Never ask the
user to paste a secret into the chat, and never put a secret value in this
tool's arguments. It is not for connecting MCP servers or OAuth sign-ins.

The value is stored in the user's secrets and exposed to this conversation as
an environment variable named after the secret. Reference it as $NAME in
commands; never print or echo it. One call per missing secret.

What comes back: this tool is acknowledged immediately, before the user has
answered, and the user answers whenever they are ready. Keep working on
anything that does not need the secret, or end your turn if nothing is left;
do not wait in a loop or ask again. The outcome arrives as a follow-up message
starting with "${SECRET_REQUEST_RESULT_PREFIX.trim()}". On "saved",
continue using the secret by the name it reports (the user may have renamed
it). Otherwise, follow the guidance in the message; do not call this tool again
unless the user asks you to.`;

export const REQUEST_SECRET_CLIENT_TOOL: ClientToolSpec = {
  name: REQUEST_SECRET_TOOL_NAME,
  description: REQUEST_SECRET_DESCRIPTION,
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      name: {
        type: "string",
        description:
          "Environment variable name for the secret, e.g. 'GITHUB_TOKEN'. Starts with a letter; letters, digits and underscores only; at most 64 characters.",
      },
      reason: {
        type: "string",
        description:
          "One sentence shown to the user explaining what the secret is for and which permissions or scopes it needs.",
      },
    },
    required: ["name", "reason"],
  },
  annotations: {
    readOnlyHint: false,
    destructiveHint: false,
    // Each call prompts the user again.
    idempotentHint: false,
    openWorldHint: false,
  },
};
