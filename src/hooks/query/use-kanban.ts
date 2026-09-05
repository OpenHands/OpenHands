import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import KanbanService from "#/api/kanban-service/kanban-service.api";
import type {
  CreateBoardPayload,
  CreateCardPayload,
  CreateColumnPayload,
  MoveCardPayload,
  UpdateCardPayload,
} from "#/api/kanban-service/kanban-types";
import { KANBAN_QUERY_KEYS } from "#/hooks/query/query-keys";

export function useKanbanBoards() {
  return useQuery({
    queryKey: KANBAN_QUERY_KEYS.boards(),
    queryFn: () => KanbanService.listBoards(),
  });
}

export function useKanbanBoard(boardId: string | null) {
  return useQuery({
    queryKey: KANBAN_QUERY_KEYS.board(boardId ?? ""),
    queryFn: () => KanbanService.getBoard(boardId!),
    enabled: Boolean(boardId),
  });
}

export function useKanbanBoardCosts(boardId: string | null) {
  return useQuery({
    queryKey: KANBAN_QUERY_KEYS.costs(boardId ?? ""),
    queryFn: () => KanbanService.getBoardCosts(boardId!),
    enabled: Boolean(boardId),
  });
}

function useInvalidateKanban(boardId?: string | null) {
  const queryClient = useQueryClient();
  return () => {
    queryClient.invalidateQueries({ queryKey: KANBAN_QUERY_KEYS.all });
    if (boardId) {
      queryClient.invalidateQueries({
        queryKey: KANBAN_QUERY_KEYS.board(boardId),
      });
      queryClient.invalidateQueries({
        queryKey: KANBAN_QUERY_KEYS.costs(boardId),
      });
    }
  };
}

export function useCreateKanbanBoard() {
  const invalidate = useInvalidateKanban();
  return useMutation({
    mutationFn: (payload: CreateBoardPayload) =>
      KanbanService.createBoard(payload),
    onSuccess: invalidate,
  });
}

export function useCreateKanbanColumn(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: (payload: CreateColumnPayload) =>
      KanbanService.addColumn(boardId, payload),
    onSuccess: invalidate,
  });
}

export function useCreateKanbanCard(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: ({
      columnId,
      payload,
    }: {
      columnId: string;
      payload: CreateCardPayload;
    }) => KanbanService.createCard(columnId, payload),
    onSuccess: invalidate,
  });
}

export function useUpdateKanbanCard(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: ({
      cardId,
      payload,
    }: {
      cardId: string;
      payload: UpdateCardPayload;
    }) => KanbanService.updateCard(cardId, payload),
    onSuccess: invalidate,
  });
}

export function useDeleteKanbanCard(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: (cardId: string) => KanbanService.deleteCard(cardId),
    onSuccess: invalidate,
  });
}

export function useMoveKanbanCard(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: ({
      cardId,
      payload,
    }: {
      cardId: string;
      payload: MoveCardPayload;
    }) => KanbanService.moveCard(cardId, payload),
    onSuccess: invalidate,
  });
}

export function useUpdateKanbanColumn(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: ({
      columnId,
      payload,
    }: {
      columnId: string;
      payload: Partial<CreateColumnPayload>;
    }) => KanbanService.updateColumn(columnId, payload),
    onSuccess: invalidate,
  });
}

export function useDeleteKanbanColumn(boardId: string) {
  const invalidate = useInvalidateKanban(boardId);
  return useMutation({
    mutationFn: (columnId: string) => KanbanService.deleteColumn(columnId),
    onSuccess: invalidate,
  });
}
