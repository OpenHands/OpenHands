import { create } from "zustand";

interface EventMessageState {
  submittedEventIds: number[]; // Avoid the flashing issue of the confirmation buttons
  v1SubmittedEventIds: string[]; // V1 event IDs (V1 uses string IDs)
}

interface EventMessageStore extends EventMessageState {
  addSubmittedEventId: (id: number) => void;
  removeSubmittedEventId: (id: number) => void;
  addV1SubmittedEventId: (id: string) => void;
  removeV1SubmittedEventId: (id: string) => void;
}

export const useEventMessageStore = create<EventMessageStore>((set) => ({
  submittedEventIds: [],
  v1SubmittedEventIds: [],
  addSubmittedEventId: (id: number) =>
    set((state) => ({
      submittedEventIds: [...state.submittedEventIds, id],
    })),
  removeSubmittedEventId: (id: number) =>
    set((state) => ({
      submittedEventIds: state.submittedEventIds.filter(
        (eventId) => eventId !== id,
      ),
    })),
  addV1SubmittedEventId: (id: string) =>
    set((state) => ({
      v1SubmittedEventIds: [...state.v1SubmittedEventIds, id],
    })),
  removeV1SubmittedEventId: (id: string) =>
    set((state) => ({
      v1SubmittedEventIds: state.v1SubmittedEventIds.filter(
        (eventId) => eventId !== id,
      ),
    })),
}));
