import { openHands } from "./open-hands-axios";

export const llmApi = openHands.create();

llmApi.defaults.baseURL = `${openHands.defaults.baseURL}/api/v1/llm`;
