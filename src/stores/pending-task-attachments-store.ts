import { create } from "zustand";

export interface PendingTaskAttachments {
  content: string;
  images: File[];
  files: File[];
  imagesMarkedUploadAsFile: string[];
}

interface PendingTaskAttachmentsState {
  byTaskId: Record<string, PendingTaskAttachments>;
  setPendingTaskAttachments: (
    taskId: string,
    payload: PendingTaskAttachments,
  ) => void;
  consumePendingTaskAttachments: (
    taskId: string,
  ) => PendingTaskAttachments | null;
}

export const usePendingTaskAttachmentsStore =
  create<PendingTaskAttachmentsState>()((set, get) => ({
    byTaskId: {},

    setPendingTaskAttachments: (taskId, payload) =>
      set((state) => ({
        byTaskId: { ...state.byTaskId, [taskId]: payload },
      })),

    consumePendingTaskAttachments: (taskId) => {
      const payload = get().byTaskId[taskId];
      if (!payload) {
        return null;
      }

      set((state) => {
        const { [taskId]: _removed, ...rest } = state.byTaskId;
        return { byTaskId: rest };
      });

      return payload;
    },
  }));

export function setPendingTaskAttachments(
  taskId: string,
  payload: PendingTaskAttachments,
): void {
  usePendingTaskAttachmentsStore
    .getState()
    .setPendingTaskAttachments(taskId, payload);
}

export function consumePendingTaskAttachments(
  taskId: string,
): PendingTaskAttachments | null {
  return usePendingTaskAttachmentsStore
    .getState()
    .consumePendingTaskAttachments(taskId);
}
