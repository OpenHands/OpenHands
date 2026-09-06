import { FILE_SERVICE_HANDLERS } from "./file-service-handlers";
import { SECRETS_HANDLERS } from "./secrets-handlers";
import {
  AGENT_PROFILES_HANDLERS,
  resetMockAgentProfiles,
  seedMockAgentProfiles,
} from "./agent-profiles-handlers";
import { GIT_REPOSITORY_HANDLERS } from "./git-repository-handlers";
import {
  SETTINGS_HANDLERS,
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
} from "./settings-handlers";
import { CONVERSATION_HANDLERS } from "./conversation-handlers";
import { AUTH_HANDLERS } from "./auth-handlers";
import { FEEDBACK_HANDLERS } from "./feedback-handlers";
import { ANALYTICS_HANDLERS } from "./analytics-handlers";
import {
  AUTOMATION_HANDLERS,
  resetAutomationMockData,
} from "./automation-handlers";
import { MCP_HANDLERS } from "./mcp-handlers";
import {
  WORKSPACES_HANDLERS,
  resetMockWorkspaces,
} from "./workspaces-handlers";
import {
  CANVAS_EXTENSIONS_HANDLERS,
  resetCanvasExtensionsMockData,
} from "./canvas-extensions-handlers";
import { KANBAN_HANDLERS, resetKanbanMockData } from "./kanban-handlers";
import { PROJECTS_HANDLERS, resetProjectsMockData } from "./projects-handlers";
import {
  FEATURE_DEV_HANDLERS,
  resetFeatureDevMockData,
} from "./feature-developer-handlers";
import { LOOP_HANDLERS, resetLoopMockData } from "./loop-handlers";
import { ROUTING_HANDLERS, resetRoutingMockData } from "./routing-handlers";
import { CHANNEL_HANDLERS, resetChannelMockData } from "./channel-handlers";
import { MEETILY_HANDLERS } from "./meetily-handlers";

export const handlers = [
  ...FILE_SERVICE_HANDLERS,
  ...SECRETS_HANDLERS,
  ...AGENT_PROFILES_HANDLERS,
  ...GIT_REPOSITORY_HANDLERS,
  ...SETTINGS_HANDLERS,
  ...CONVERSATION_HANDLERS,
  ...AUTH_HANDLERS,
  ...FEEDBACK_HANDLERS,
  ...ANALYTICS_HANDLERS,
  ...AUTOMATION_HANDLERS,
  ...MCP_HANDLERS,
  ...WORKSPACES_HANDLERS,
  ...CANVAS_EXTENSIONS_HANDLERS,
  ...KANBAN_HANDLERS,
  ...PROJECTS_HANDLERS,
  ...FEATURE_DEV_HANDLERS,
  ...LOOP_HANDLERS,
  ...ROUTING_HANDLERS,
  ...CHANNEL_HANDLERS,
  ...MEETILY_HANDLERS,
];

export {
  MOCK_DEFAULT_USER_SETTINGS,
  resetTestHandlersMockSettings,
  resetAutomationMockData,
  resetMockWorkspaces,
  resetCanvasExtensionsMockData,
  resetKanbanMockData,
  resetProjectsMockData,
  resetFeatureDevMockData,
  resetLoopMockData,
  resetRoutingMockData,
  resetChannelMockData,
};

export {
  AGENT_PROFILES_HANDLERS,
  resetMockAgentProfiles,
  seedMockAgentProfiles,
};
