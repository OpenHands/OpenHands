import axios from "axios";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { getEffectiveLocalBackend } from "../backend-registry/active-store";
import {
  KANBAN_API_BOARDS_PATH,
  KANBAN_API_CARDS_PATH,
  KANBAN_API_COLUMNS_PATH,
  PROJECT_API_INIT_PATH,
  PROJECT_API_PREVIEW_PATH,
  SESSION_API_KEY_HEADER,
} from "./kanban-constants";
import type {
  CreateBoardPayload,
  CreateCardPayload,
  CreateColumnPayload,
  KanbanBoard,
  KanbanBoardCosts,
  KanbanBoardSummary,
  KanbanCard,
  KanbanColumn,
  MoveCardPayload,
  ProjectInitPayload,
  ProjectInitResult,
  ProjectPreviewResult,
  UpdateCardPayload,
} from "./kanban-types";

const kanbanAxios = axios.create();

kanbanAxios.interceptors.request.use((config) => {
  const backend = getEffectiveLocalBackend();
  if (!backend) throw new NoBackendAvailableError();
  // eslint-disable-next-line no-param-reassign
  config.baseURL = backend.host;
  const apiKey = backend.apiKey?.trim();
  if (apiKey) {
    config.headers.set(SESSION_API_KEY_HEADER, apiKey);
  }
  return config;
});

export const KanbanService = {
  listBoards: async (): Promise<KanbanBoardSummary[]> => {
    const { data } = await kanbanAxios.get<KanbanBoardSummary[]>(
      KANBAN_API_BOARDS_PATH,
    );
    return data;
  },

  createBoard: async (payload: CreateBoardPayload): Promise<KanbanBoard> => {
    const { data } = await kanbanAxios.post<KanbanBoard>(
      KANBAN_API_BOARDS_PATH,
      payload,
    );
    return data;
  },

  getBoard: async (boardId: string): Promise<KanbanBoard> => {
    const { data } = await kanbanAxios.get<KanbanBoard>(
      `${KANBAN_API_BOARDS_PATH}/${boardId}`,
    );
    return data;
  },

  getBoardCosts: async (boardId: string): Promise<KanbanBoardCosts> => {
    const { data } = await kanbanAxios.get<KanbanBoardCosts>(
      `${KANBAN_API_BOARDS_PATH}/${boardId}/costs`,
    );
    return data;
  },

  addColumn: async (
    boardId: string,
    payload: CreateColumnPayload,
  ): Promise<KanbanColumn> => {
    const { data } = await kanbanAxios.post<KanbanColumn>(
      `${KANBAN_API_BOARDS_PATH}/${boardId}/columns`,
      payload,
    );
    return data;
  },

  updateColumn: async (
    columnId: string,
    payload: Partial<CreateColumnPayload>,
  ): Promise<KanbanColumn> => {
    const { data } = await kanbanAxios.patch<KanbanColumn>(
      `${KANBAN_API_COLUMNS_PATH}/${columnId}`,
      payload,
    );
    return data;
  },

  deleteColumn: async (columnId: string): Promise<void> => {
    await kanbanAxios.delete(`${KANBAN_API_COLUMNS_PATH}/${columnId}`);
  },

  createCard: async (
    columnId: string,
    payload: CreateCardPayload,
  ): Promise<KanbanCard> => {
    const { data } = await kanbanAxios.post<KanbanCard>(
      `${KANBAN_API_COLUMNS_PATH}/${columnId}/cards`,
      payload,
    );
    return data;
  },

  updateCard: async (
    cardId: string,
    payload: UpdateCardPayload,
  ): Promise<KanbanCard> => {
    const { data } = await kanbanAxios.patch<KanbanCard>(
      `${KANBAN_API_CARDS_PATH}/${cardId}`,
      payload,
    );
    return data;
  },

  deleteCard: async (cardId: string): Promise<void> => {
    await kanbanAxios.delete(`${KANBAN_API_CARDS_PATH}/${cardId}`);
  },

  moveCard: async (
    cardId: string,
    payload: MoveCardPayload,
  ): Promise<KanbanCard> => {
    const { data } = await kanbanAxios.post<KanbanCard>(
      `${KANBAN_API_CARDS_PATH}/${cardId}/move`,
      payload,
    );
    return data;
  },

  linkSession: async (
    cardId: string,
    sessionId: string,
  ): Promise<KanbanCard> => {
    const { data } = await kanbanAxios.post<KanbanCard>(
      `${KANBAN_API_CARDS_PATH}/${cardId}/link-session`,
      { session_id: sessionId },
    );
    return data;
  },

  previewProject: async (
    payload: ProjectInitPayload,
  ): Promise<ProjectPreviewResult> => {
    const { data } = await kanbanAxios.post<ProjectPreviewResult>(
      PROJECT_API_PREVIEW_PATH,
      payload,
    );
    return data;
  },

  initProject: async (
    payload: ProjectInitPayload,
  ): Promise<ProjectInitResult> => {
    const { data } = await kanbanAxios.post<ProjectInitResult>(
      PROJECT_API_INIT_PATH,
      payload,
    );
    return data;
  },
};

export default KanbanService;
