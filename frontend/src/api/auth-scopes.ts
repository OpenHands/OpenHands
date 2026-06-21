import { openHands } from "./open-hands-axios";

export interface ScopeInfo {
  name: string;
  description: string;
  is_default: boolean;
  is_visible_to_users: boolean;
}

export const getAuthScopes = async (): Promise<ScopeInfo[]> => {
  const { data } = await openHands.get<ScopeInfo[]>(
    "/api/v1/users/auth/scopes",
  );
  return data;
};
