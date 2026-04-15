import { openHands } from "../open-hands-axios";
import { GitUser } from "#/types/git";
import { UserGitOrganizationsResponse } from "#/types/org";

/**
 * User Service API - Handles all user-related API endpoints
 */
class UserService {
  /**
   * Get the current user's Git information
   * @returns Git user information
   */
  static async getUser(): Promise<GitUser> {
    const { data } = await openHands.get<GitUser>("/api/v1/users/git-info");
    return data;
  }

  /**
   * Get the current user's Git organizations
   * @returns Git organizations for the current user's provider
   */
  static async getGitOrganizations(): Promise<UserGitOrganizationsResponse> {
    const { data } = await openHands.get<UserGitOrganizationsResponse>(
      "/api/user/git-organizations",
    );
    return data;
  }
}

export default UserService;
