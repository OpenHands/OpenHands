import { create } from "zustand";

type SubmittedEventId = string | number;

interface EventMessageState {
  submittedEventIds: SubmittedEventId[];
}

interface EventMessageStore extends EventMessageState {
  addSubmittedEventId: (id: SubmittedEventId) => void;
  removeSubmittedEventId: (id: SubmittedEventId) => void;
}

export const useEventMessageStore = create<EventMessageStore>((set) => ({
  submittedEventIds: [],
  addSubmittedEventId: (id) =>
    set((state) => ({
      submittedEventIds: [...state.submittedEventIds, id],
    })),
  removeSubmittedEventId: (id) =>
    set((state) => ({
      submittedEventIds: state.submittedEventIds.filter(
        (eventId) => eventId !== id,
      ),
    })),
}));
