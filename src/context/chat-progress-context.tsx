import { createContext, useContext } from "react";

export const ChatProgressContext = createContext(false);

export const useChatProgress = () => useContext(ChatProgressContext);
