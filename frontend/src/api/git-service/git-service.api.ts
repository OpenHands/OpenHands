import { openHands } from "../open-hands-axios";
import { RepositoryPage, BranchPage, InstallationPage } from "#/types/git";
import { GitChange, GitChangeDiff } from "../open-hands.types";
import ConversationService from "../conversation-service/conversation-service.api";

/**
 * Git Service API - Handles all Git-related API endpoints
 */
class GitService {
  /**
   * Search for Git repositories
   * @param query Search query
   * @param provider Git provider to search in (required)
   * @param limit Number of results per page
   * @param pageId Cursor for pagination
   * @param installationId Filter by installation ID
   * @param sortOrder Sort order (asc or desc)
   * @returns Paginated repository response
   */
  static async searchGitRepositories(
    query: string,
    provider: string,
    limit = 100,
    pageId?: string,
    installationId?: string,
  ): Promise<RepositoryPage> {
    const { data } = await openHands.get<RepositoryPage>(
      "/api/v1/git/repositories/search",
      {
        params: {
          provider,
          query,
          limit,
          page_id: pageId,
          installation_id: installationId,
        },
      },
    );

    return data;
  }

  /**
   * Retrieve user's Git repositories
   * @param provider Git provider
   * @param pageId Cursor for pagination
   * @param limit Number of results per page
   * @param installationId Filter by installation ID
   * @param sortOrder Sort order (asc or desc)
   * @returns User's repositories with pagination info
   */
  static async retrieveUserGitRepositories(
    provider: string,
    pageId?: string,
    limit = 30,
    installationId?: string,
  ): Promise<RepositoryPage> {
    const { data } = await openHands.get<RepositoryPage>(
      "/api/v1/git/repositories/search",
      {
        params: {
          provider,
          limit,
          page_id: pageId,
          installation_id: installationId,
        },
      },
    );

    return data;
  }

  /**
   * Retrieve repositories from a specific installation
   * @param provider Git provider
   * @param installationIndex Current installation index
   * @param installations List of installation IDs
   * @param pageId Cursor for pagination
   * @param limit Number of results per page
   * @returns Installation repositories with pagination info
   */
  static async retrieveInstallationRepositories(
    provider: string,
    installationIndex: number,
    installations: string[],
    pageId?: string,
    limit = 30,
  ): Promise<RepositoryPage> {
    const installationId = installations[installationIndex];
    const { data } = await openHands.get<RepositoryPage>(
      "/api/v1/git/repositories/search",
      {
        params: {
          provider,
          limit,
          page_id: pageId,
          installation_id: installationId,
        },
      },
    );
    return data;
  }

  /**
   * Get repository branches
   * @param repository Repository name
   * @param provider Git provider (required)
   * @param query Search query (required - can be empty string)
   * @param pageId Cursor for pagination
   * @param limit Number of results per page
   * @returns Paginated branches response
   */
  static async getRepositoryBranches(
    repository: string,
    provider: string,
    query: string = "",
    pageId?: string,
    limit = 30,
  ): Promise<BranchPage> {
    const { data } = await openHands.get<BranchPage>(
      "/api/v1/git/branches/search",
      {
        params: {
          provider,
          repository,
          query,
          page_id: pageId,
          limit,
        },
      },
    );

    return data;
  }

  /**
   * Search repository branches
   * @deprecated Use getRepositoryBranches instead - this method is identical
   * @param repository Repository name
   * @param provider Git provider (required)
   * @param query Search query
   * @param pageId Cursor for pagination
   * @param limit Number of results per page
   * @returns List of matching branches
   */
  static async searchRepositoryBranches(
    repository: string,
    provider: string,
    query: string,
    pageId?: string,
    limit = 30,
  ): Promise<BranchPage> {
    return this.getRepositoryBranches(
      repository,
      provider,
      query,
      pageId,
      limit,
    );
  }

  /**
   * Get the user installation IDs
   * @param provider The provider to get installation IDs for (github, bitbucket, etc.)
   * @param pageId Cursor for pagination
   * @param limit Max number of results
   * @returns Paginated installation response
   */
  static async getUserInstallations(
    provider: string,
    pageId?: string,
    limit = 100,
  ): Promise<InstallationPage> {
    const { data } = await openHands.get<InstallationPage>(
      "/api/v1/git/installations/search",
      {
        params: {
          provider,
          page_id: pageId,
          limit,
        },
      },
    );
    return data;
  }

  /**
   * Get git changes for a conversation
   * @param conversationId The conversation ID
   * @returns List of git changes
   */
  static async getGitChanges(conversationId: string): Promise<GitChange[]> {
    const url = `${ConversationService.getConversationUrl(conversationId)}/git/changes`;
    const { data } = await openHands.get<GitChange[]>(url, {
      headers: ConversationService.getConversationHeaders(),
    });
    return data;
  }

  /**
   * Get git change diff for a specific file
   * @param conversationId The conversation ID
   * @param path The file path
   * @returns Git change diff
   */
  static async getGitChangeDiff(
    conversationId: string,
    path: string,
  ): Promise<GitChangeDiff> {
    const url = `${ConversationService.getConversationUrl(conversationId)}/git/diff`;
    const { data } = await openHands.get<GitChangeDiff>(url, {
      params: { path },
      headers: ConversationService.getConversationHeaders(),
    });
    return data;
  }
}

export default GitService;
