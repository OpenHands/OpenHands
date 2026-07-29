import { create } from "zustand";
import { StatusMessage } from "#/types/message";

const initialStatusMessage: StatusMessage = {
  status_update: true,
  type: "info",
  id: "",
  message: "",
};

interface StatusState {
  curStatusMessage: StatusMessage;
  setCurStatusMessage: (message: StatusMessage) => void;
}

export const useStatusStore = create<StatusState>((set) => ({
  curStatusMessage: initialStatusMessage,
  setCurStatusMessage: (message: StatusMessage) =>
    set({ curStatusMessage: message }),
}));
