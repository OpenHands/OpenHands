import { openHands } from "../open-hands-axios";
import { Provider } from "#/types/settings";
import { GitRepository, PaginatedBranchesResponse, Branch } from "#/types/git";
import { extractNextPageFromLink } from "#/utils/extract-next-page-from-link";
import { RepositoryMicroagent } from "#/types/microagent-management";
import {
  MicroagentContentResponse,
  GitChange,
  GitChangeDiff,
} from "../open-hands.types";
import ConversationService from "../conversation-service/conversation-service.api";

/**
 * Git Service API - Handles all Git-related API endpoints
 */
class GitService {
  /**
   * Search for Git repositories
   * @param query Search query
   * @param per_page Number of results per page
   * @param selected_provider Git provider to search in
   * @returns List of matching repositories
   */
  static async searchGitRepositories(
    query: string,
    per_page = 100,
    selected_provider?: Provider,
  ): Promise<GitRepository[]> {
    const response = await openHands.get<GitRepository[]>(
      "/api/user/search/repositories",
      {
        params: {
          query,
          per_page,
          selected_provider,
        },
      },
    );

    return response.data;
  }

  /**
   * Retrieve user's Git repositories
   * @param selected_provider Git provider
   * @param page Page number
   * @param per_page Number of results per page
   * @returns User's repositories with pagination info
   */
  static async retrieveUserGitRepositories(
    selected_provider: Provider,
    page = 1,
    per_page = 30,
  ) {
    const { data } = await openHands.get<GitRepository[]>(
      "/api/user/repositories",
      {
        params: {
          selected_provider,
          sort: "pushed",
          page,
          per_page,
        },
      },
    );

    const link =
      data.length > 0 && data[0].link_header ? data[0].link_header : "";
    const nextPage = extractNextPageFromLink(link);

    return { data, nextPage };
  }

  /**
   * Retrieve repositories from a specific installation
   * @param selected_provider Git provider
   * @param installationIndex Current installation index
   * @param installations List of installation IDs
   * @param page Page number
   * @param per_page Number of results per page
   * @returns Installation repositories with pagination info
   */
  static async retrieveInstallationRepositories(
    selected_provider: Provider,
    installationIndex: number,
    installations: string[],
    page = 1,
    per_page = 30,
  ) {
    const installationId = installations[installationIndex];
    const response = await openHands.get<GitRepository[]>(
      "/api/user/repositories",
      {
        params: {
          selected_provider,
          sort: "pushed",
          page,
          per_page,
          installation_id: installationId,
        },
      },
    );
    const link =
      response.data.length > 0 && response.data[0].link_header
        ? response.data[0].link_header
        : "";
    const nextPage = extractNextPageFromLink(link);
    let nextInstallation: number | null;
    if (nextPage) {
      nextInstallation = installationIndex;
    } else if (installationIndex + 1 < installations.length) {
      nextInstallation = installationIndex + 1;
    } else {
      nextInstallation = null;
    }
    return {
      data: response.data,
      nextPage,
      installationIndex: nextInstallation,
    };
  }

  /**
   * Get repository branches
   * @param repository Repository name
   * @param page Page number
   * @param perPage Number of results per page
   * @returns Paginated branches response
   */
  static async getRepositoryBranches(
    repository: string,
    page: number = 1,
    perPage: number = 30,
    selectedProvider?: Provider,
  ): Promise<PaginatedBranchesResponse> {
    const { data } = await openHands.get<PaginatedBranchesResponse>(
      `/api/user/repository/branches`,
      {
        params: {
          repository,
          page,
          per_page: perPage,
          selected_provider: selectedProvider,
        },
      },
    );

    return data;
  }

  /**
   * Search repository branches
   * @param repository Repository name
   * @param query Search query
   * @param perPage Number of results per page
   * @param selectedProvider Git provider
   * @returns List of matching branches
   */
  static async searchRepositoryBranches(
    repository: string,
    query: string,
    perPage: number = 30,
    selectedProvider?: Provider,
  ): Promise<Branch[]> {
    const { data } = await openHands.get<Branch[]>(
      `/api/user/search/branches`,
      {
        params: {
          repository,
          query,
          per_page: perPage,
          selected_provider: selectedProvider,
        },
      },
    );
    return data;
  }

  /**
   * Get the available microagents for a repository
   * @param owner The repository owner
   * @param repo The repository name
   * @returns The available microagents for the repository
   */
  static async getRepositoryMicroagents(
    owner: string,
    repo: string,
  ): Promise<RepositoryMicroagent[]> {
    const { data } = await openHands.get<RepositoryMicroagent[]>(
      `/api/user/repository/${owner}/${repo}/microagents`,
    );
    return data;
  }

  /**
   * Get the content of a specific microagent from a repository
   * @param owner The repository owner
   * @param repo The repository name
   * @param filePath The path to the microagent file within the repository
   * @returns The microagent content and metadata
   */
  static async getRepositoryMicroagentContent(
    owner: string,
    repo: string,
    filePath: string,
  ): Promise<MicroagentContentResponse> {
    const { data } = await openHands.get<MicroagentContentResponse>(
      `/api/user/repository/${owner}/${repo}/microagents/content`,
      {
        params: { file_path: filePath },
      },
    );
    return data;
  }

  /**
   * Get the user installation IDs
   * @param provider The provider to get installation IDs for (github, bitbucket, etc.)
   * @returns List of installation IDs
   */
  static async getUserInstallationIds(provider: Provider): Promise<string[]> {
    const { data } = await openHands.get<string[]>(
      `/api/user/installations?provider=${provider}`,
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
