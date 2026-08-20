// Separate route module so React Router does not see duplicate ids for
// `conversations/:id` and `conversations/:id/panel` (same handlers as conversation).
export { default, ConversationView } from "./conversation";
