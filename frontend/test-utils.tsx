// Test utilities for React components

import React, { PropsWithChildren } from "react";
import { RenderOptions, render } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { I18nextProvider, initReactI18next } from "react-i18next";
import i18n from "i18next";
import { vi } from "vitest";
import { AxiosError } from "axios";
import {
  ActionEvent,
  MessageEvent,
  ObservationEvent,
  PlanningFileEditorObservation,
} from "#/types/v1/core";
import { SecurityRisk } from "#/types/v1/core";

export const useParamsMock = vi.fn(() => ({
  conversationId: "test-conversation-id",
}));

// Mock useParams before importing components
vi.mock("react-router", async () => {
  const actual =
    await vi.importActual<typeof import("react-router")>("react-router");
  return {
    ...actual,
    useParams: useParamsMock,
  };
});

// Initialize i18n for tests
i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  ns: ["translation"],
  defaultNS: "translation",
  resources: {
    en: {
      translation: {},
    },
  },
  interpolation: {
    escapeValue: false,
  },
});

// This type interface extends the default options for render from RTL
interface ExtendedRenderOptions extends Omit<RenderOptions, "queries"> {}

// Export our own customized renderWithProviders function that renders with QueryClient and i18next providers
// Since we're using Zustand stores, we don't need a Redux Provider wrapper
export function renderWithProviders(
  ui: React.ReactElement,
  renderOptions: ExtendedRenderOptions = {},
) {
  function Wrapper({ children }: PropsWithChildren) {
    return (
      <QueryClientProvider
        client={
          new QueryClient({
            defaultOptions: { queries: { retry: false } },
          })
        }
      >
        <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
      </QueryClientProvider>
    );
  }
  return render(ui, { wrapper: Wrapper, ...renderOptions });
}

export const createAxiosNotFoundErrorObject = () =>
  new AxiosError(
    "Request failed with status code 404",
    "ERR_BAD_REQUEST",
    undefined,
    undefined,
    {
      status: 404,
      statusText: "Not Found",
      data: { message: "Settings not found" },
      headers: {},
      // @ts-expect-error - we only need the response object for this test
      config: {},
    },
  );

export const createAxiosError = (
  status: number,
  statusText: string,
  data: unknown,
) =>
  new AxiosError(
    `Request failed with status code ${status}`,
    "ERR_BAD_REQUEST",
    undefined,
    undefined,
    {
      status,
      statusText,
      data,
      headers: {},
      // @ts-expect-error - we only need the response object for this test
      config: {},
    },
  );

// Helper to create a PlanningFileEditorAction event
export const createPlanningFileEditorActionEvent = (
  id: string,
): ActionEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Planning action" }],
  thinking_blocks: [],
  action: {
    kind: "PlanningFileEditorAction",
    command: "create",
    path: "/workspace/PLAN.md",
    file_text: "Plan content",
    old_str: null,
    new_str: null,
    insert_line: null,
    view_range: null,
  },
  tool_name: "planning_file_editor",
  tool_call_id: "call-1",
  tool_call: {
    id: "call-1",
    type: "function",
    function: {
      name: "planning_file_editor",
      arguments: '{"command": "create"}',
    },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.UNKNOWN,
});

// Helper to create a non-planning action event
export const createOtherActionEvent = (id: string): ActionEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought: [{ type: "text", text: "Other action" }],
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: "echo test",
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: "call-1",
  tool_call: {
    id: "call-1",
    type: "function",
    function: {
      name: "execute_bash",
      arguments: '{"command": "echo test"}',
    },
  },
  llm_response_id: "response-1",
  security_risk: SecurityRisk.UNKNOWN,
});

// Helper to create a PlanningFileEditorObservation event
export const createPlanningObservationEvent = (
  id: string,
  actionId: string = "action-1",
): ObservationEvent<PlanningFileEditorObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "planning_file_editor",
  tool_call_id: "call-1",
  action_id: actionId,
  observation: {
    kind: "PlanningFileEditorObservation",
    content: [{ type: "text", text: "Plan content" }],
    is_error: false,
    command: "create",
    path: "/workspace/PLAN.md",
    prev_exist: false,
    old_content: null,
    new_content: "Plan content",
  },
});

// Helper to create a user message event
export const createUserMessageEvent = (id: string): MessageEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "User message" }],
  },
  activated_microagents: [],
  extended_content: [],
});
