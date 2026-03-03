import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { Provider } from "#/types/settings";
import { SuggestedTaskGroup } from "#/utils/types";
import { ConversationStatus } from "#/types/conversation-status";
import { GitRepository } from "#/types/git";
import { sanitizeQuery } from "#/utils/sanitize-query";
import { PRODUCT_URL } from "#/utils/constants";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Trigger a download for a provided Blob with the given filename
 */
export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

/**
 * Get the numeric height value from an element's style property
 * @param el The HTML element to get the height from
 * @param fallback The fallback value to return if style height is invalid
 * @returns The numeric height value in pixels, or the fallback value
 *
 * @example
 * getStyleHeightPx(element, 20) // Returns 20 if element.style.height is "auto" or invalid
 * getStyleHeightPx(element, 20) // Returns 100 if element.style.height is "100px"
 */
export const getStyleHeightPx = (el: HTMLElement, fallback: number): number => {
  const elementHeight = parseFloat(el.style.height || "");
  return Number.isFinite(elementHeight) ? elementHeight : fallback;
};

/**
 * Set the height style property of an element to a specific pixel value
 * @param el The HTML element to set the height for
 * @param height The height value in pixels to set
 *
 * @example
 * setStyleHeightPx(element, 100) // Sets element.style.height to "100px"
 * setStyleHeightPx(textarea, 200) // Sets textarea.style.height to "200px"
 */
export const setStyleHeightPx = (el: HTMLElement, height: number): void => {
  el.style.setProperty("height", `${height}px`);
};

/**
 * Detect if the user is on a mobile device.
 * Touch support alone is not sufficient — touchscreen laptops have touch
 * but use a mouse/trackpad as primary input. We check that the primary
 * pointing device is coarse (finger) to avoid false positives.
 */
export const isMobileDevice = (): boolean => {
  if (
    /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    )
  )
    return true;

  const hasTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
  if (!hasTouch) return false;

  // If matchMedia is available, check whether the primary pointer is fine
  // (mouse/trackpad). Touchscreen laptops report fine, real mobile devices don't.
  if (typeof window.matchMedia === "function") {
    return !window.matchMedia("(pointer: fine)").matches;
  }

  // Fallback: touch present but no matchMedia — assume mobile
  return true;
};

/**
 * Checks if the current domain is the production domain
 * @returns True if the current domain matches the production URL
 */
export const isProductionDomain = (): boolean =>
  window.location.origin === PRODUCT_URL.PRODUCTION;

interface EventActionHistory {
  args?: {
    LLM_API_KEY?: string;
    [key: string]: unknown;
  };
  extras?: {
    open_page_urls: string[];
    active_page_index: number;
    dom_object: Record<string, unknown>;
    axtree_object: Record<string, unknown>;
    extra_element_properties: Record<string, unknown>;
    last_browser_action: string;
    last_browser_action_error: unknown;
    focused_element_bid: string;
  };
  [key: string]: unknown;
}

export const removeUnwantedKeys = (
  data: EventActionHistory[],
): EventActionHistory[] => {
  const UNDESIRED_KEYS = [
    "open_page_urls",
    "active_page_index",
    "dom_object",
    "axtree_object",
    "extra_element_properties",
    "last_browser_action",
    "last_browser_action_error",
    "focused_element_bid",
  ];

  return data
    .filter((item) => {
      // Skip items that have a status key
      if ("status" in item) {
        return false;
      }
      return true;
    })
    .map((item) => {
      // Create a shallow copy of item
      const newItem = { ...item };

      // Check if extras exists and delete it from a new extras object
      if (newItem.extras) {
        const newExtras = { ...newItem.extras };
        UNDESIRED_KEYS.forEach((key) => {
          delete newExtras[key as keyof typeof newExtras];
        });
        newItem.extras = newExtras;
      }

      return newItem;
    });
};

export const removeApiKey = (
  data: EventActionHistory[],
): EventActionHistory[] =>
  data.map((item) => {
    // Create a shallow copy of item
    const newItem = { ...item };

    // Check if LLM_API_KEY exists and delete it from a new args object
    if (newItem.args?.LLM_API_KEY) {
      const newArgs = { ...newItem.args };
      delete newArgs.LLM_API_KEY;
      newItem.args = newArgs;
    }

    return newItem;
  });

export const getExtension = (code: string) => {
  if (code.includes(".")) return code.split(".").pop() || "";
  return "";
};

/**
 * Get file extension from file name in uppercase format
 * @param fileName The file name to extract extension from
 * @returns The file extension in uppercase, or "FILE" if no extension found
 *
 * @example
 * getFileExtension("document.pdf") // "PDF"
 * getFileExtension("image.jpeg") // "JPEG"
 * getFileExtension("noextension") // "FILE"
 */
export const getFileExtension = (fileName: string): string => {
  const extension = fileName.split(".").pop()?.toUpperCase();
  return extension || "FILE";
};

/**
 * Format a timestamp to a human-readable format
 * @param timestamp The timestamp to format (ISO 8601)
 * @returns The formatted timestamp
 *
 * @example
 * formatTimestamp("2021-10-10T10:10:10.000") // "10/10/2021, 10:10:10"
 * formatTimestamp("2021-10-10T22:10:10.000") // "10/10/2021, 22:10:10"
 */
export const formatTimestamp = (timestamp: string) =>
  new Date(timestamp).toLocaleString("en-GB", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

export const shouldUseInstallationRepos = (
  provider: Provider,
  app_mode: "saas" | "oss" | undefined,
) => {
  if (!provider) return false;

  switch (provider) {
    case "bitbucket":
      return true;
    case "gitlab":
      return false;
    case "azure_devops":
      return false;
    case "github":
      return app_mode === "saas";
    default:
      return false;
  }
};

export const getGitProviderBaseUrl = (gitProvider: Provider): string => {
  switch (gitProvider) {
    case "github":
      return "https://github.com";
    case "gitlab":
      return "https://gitlab.com";
    case "bitbucket":
      return "https://bitbucket.org";
    case "azure_devops":
      return "https://dev.azure.com";
    case "forgejo":
      // Default UI links to Codeberg unless a custom host is available in settings
      // Note: UI link builders don't currently receive host; consider plumbing settings if needed
      return "https://codeberg.org";
    default:
      return "";
  }
};

/**
 * Get the name of the git provider
 * @param gitProvider The git provider
 * @returns The name of the git provider
 */
export const getProviderName = (gitProvider: Provider) => {
  if (gitProvider === "gitlab") return "GitLab";
  if (gitProvider === "bitbucket") return "Bitbucket";
  if (gitProvider === "azure_devops") return "Azure DevOps";
  if (gitProvider === "forgejo") return "Forgejo";
  return "GitHub";
};

/**
 * Get the name of the PR
 * @param isGitLab Whether the git provider is GitLab
 * @returns The name of the PR
 */
export const getPR = (isGitLab: boolean) =>
  isGitLab ? "merge request" : "pull request";

/**
 * Get the short name of the PR
 * @param isGitLab Whether the git provider is GitLab
 * @returns The short name of the PR
 */
export const getPRShort = (isGitLab: boolean) => (isGitLab ? "MR" : "PR");

/**
 * Construct the pull request (merge request) URL for different providers
 * @param prNumber The pull request number
 * @param provider The git provider
 * @param repositoryName The repository name in format "owner/repo"
 * @returns The pull request URL
 *
 * @example
 * constructPullRequestUrl(123, "github", "owner/repo") // "https://github.com/owner/repo/pull/123"
 * constructPullRequestUrl(456, "gitlab", "owner/repo") // "https://gitlab.com/owner/repo/-/merge_requests/456"
 * constructPullRequestUrl(789, "bitbucket", "owner/repo") // "https://bitbucket.org/owner/repo/pull-requests/789"
 */
export const constructPullRequestUrl = (
  prNumber: number,
  provider: Provider,
  repositoryName: string,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider);

  switch (provider) {
    case "github":
      return `${baseUrl}/${repositoryName}/pull/${prNumber}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/pull/${prNumber}`;
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/merge_requests/${prNumber}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/pull-requests/${prNumber}`;
    case "azure_devops": {
      // Azure DevOps format: org/project/repo
      const parts = repositoryName.split("/");
      if (parts.length === 3) {
        const [org, project, repo] = parts;
        return `${baseUrl}/${org}/${project}/_git/${repo}/pullrequest/${prNumber}`;
      }
      return "";
    }
    default:
      return "";
  }
};

/**
 * Construct the microagent URL for different providers
 * @param gitProvider The git provider
 * @param repositoryName The repository name in format "owner/repo"
 * @param microagentPath The path to the microagent in the repository
 * @returns The URL to the microagent file in the Git provider
 *
 * @example
 * constructMicroagentUrl("github", "owner/repo", ".openhands/microagents/tell-me-a-joke.md")
 * // "https://github.com/owner/repo/blob/main/.openhands/microagents/tell-me-a-joke.md"
 * constructMicroagentUrl("gitlab", "owner/repo", "microagents/git-helper.md")
 * // "https://gitlab.com/owner/repo/-/blob/main/microagents/git-helper.md"
 * constructMicroagentUrl("bitbucket", "owner/repo", ".openhands/microagents/docker-helper.md")
 * // "https://bitbucket.org/owner/repo/src/main/.openhands/microagents/docker-helper.md"
 */
export const constructMicroagentUrl = (
  gitProvider: Provider,
  repositoryName: string,
  microagentPath: string,
): string => {
  const baseUrl = getGitProviderBaseUrl(gitProvider);

  switch (gitProvider) {
    case "github":
      return `${baseUrl}/${repositoryName}/blob/main/${microagentPath}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/src/branch/main/${microagentPath}`;
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/blob/main/${microagentPath}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/src/main/${microagentPath}`;
    case "azure_devops": {
      // Azure DevOps format: org/project/repo
      const parts = repositoryName.split("/");
      if (parts.length === 3) {
        const [org, project, repo] = parts;
        return `${baseUrl}/${org}/${project}/_git/${repo}?path=/${microagentPath}&version=GBmain`;
      }
      return "";
    }
    default:
      return "";
  }
};

/**
 * Extract repository owner, repo name, and file path from repository and microagent data
 * @param selectedRepository The selected repository object with full_name property
 * @param microagent The microagent object with path property
 * @returns Object containing owner, repo, and filePath
 *
 * @example
 * const { owner, repo, filePath } = extractRepositoryInfo(selectedRepository, microagent);
 */
export const extractRepositoryInfo = (
  selectedRepository: { full_name?: string } | null | undefined,
  microagent: { path?: string } | null | undefined,
) => {
  const [owner, repo] = selectedRepository?.full_name?.split("/") || [];
  const filePath = microagent?.path || "";

  return { owner, repo, filePath };
};

/**
 * Construct the repository URL for different providers
 * @param provider The git provider
 * @param repositoryName The repository name in format "owner/repo"
 * @returns The repository URL
 *
 * @example
 * constructRepositoryUrl("github", "owner/repo") // "https://github.com/owner/repo"
 * constructRepositoryUrl("gitlab", "owner/repo") // "https://gitlab.com/owner/repo"
 * constructRepositoryUrl("bitbucket", "owner/repo") // "https://bitbucket.org/owner/repo"
 */
export const constructRepositoryUrl = (
  provider: Provider,
  repositoryName: string,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider);
  return `${baseUrl}/${repositoryName}`;
};

/**
 * Construct the branch URL for different providers
 * @param provider The git provider
 * @param repositoryName The repository name in format "owner/repo"
 * @param branchName The branch name
 * @returns The branch URL
 *
 * @example
 * constructBranchUrl("github", "owner/repo", "main") // "https://github.com/owner/repo/tree/main"
 * constructBranchUrl("gitlab", "owner/repo", "develop") // "https://gitlab.com/owner/repo/-/tree/develop"
 * constructBranchUrl("bitbucket", "owner/repo", "feature") // "https://bitbucket.org/owner/repo/src/feature"
 */
export const constructBranchUrl = (
  provider: Provider,
  repositoryName: string,
  branchName: string,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider);

  switch (provider) {
    case "github":
      return `${baseUrl}/${repositoryName}/tree/${branchName}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/src/branch/${branchName}`;
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/tree/${branchName}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/src/${branchName}`;
    case "azure_devops": {
      // Azure DevOps format: org/project/repo
      const parts = repositoryName.split("/");
      if (parts.length === 3) {
        const [org, project, repo] = parts;
        return `${baseUrl}/${org}/${project}/_git/${repo}?version=GB${branchName}`;
      }
      return "";
    }
    default:
      return "";
  }
};

// Git Action Prompts

/**
 * Generate a git pull prompt
 * @returns The git pull prompt
 */
export const getGitPullPrompt = (): string =>
  "Please pull the latest code from the repository.";

/**
 * Generate a git push prompt
 * @param gitProvider The git provider
 * @returns The git push prompt
 */
export const getGitPushPrompt = (gitProvider: Provider): string => {
  const providerName = getProviderName(gitProvider);
  const pr = getPR(gitProvider === "gitlab");

  return `Please push the changes to a remote branch on ${providerName}, but do NOT create a ${pr}. Check your current branch name first - if it's main, master, deploy, or another common default branch name, create a new branch with a descriptive name related to your changes. Otherwise, use the exact SAME branch name as the one you are currently on.`;
};

/**
 * Generate a create pull request prompt
 * @param gitProvider The git provider
 * @returns The create PR prompt
 */
export const getCreatePRPrompt = (gitProvider: Provider): string => {
  const providerName = getProviderName(gitProvider);
  const pr = getPR(gitProvider === "gitlab");
  const prShort = getPRShort(gitProvider === "gitlab");

  return `Please push the changes to ${providerName} and open a ${pr}. If you're on a default branch (e.g., main, master, deploy), create a new branch with a descriptive name otherwise use the current branch. If a ${pr} template exists in the repository, please follow it when creating the ${prShort} description.`;
};

/**
 * Generate a push to existing PR prompt
 * @param gitProvider The git provider
 * @returns The push to PR prompt
 */
export const getPushToPRPrompt = (gitProvider: Provider): string => {
  const pr = getPR(gitProvider === "gitlab");

  return `Please push the latest changes to the existing ${pr}.`;
};

/**
 * Generate a create new branch prompt
 * @returns The create new branch prompt
 */
export const getCreateNewBranchPrompt = (): string =>
  "Please create a new branch with a descriptive name related to the work you plan to do.";

// Helper functions
export function getTotalTaskCount(
  suggestedTasks: SuggestedTaskGroup[] | undefined,
): number {
  if (!suggestedTasks) return 0;
  return suggestedTasks.flatMap((group) => group.tasks).length;
}

export function getLimitedTaskGroups(
  suggestedTasks: SuggestedTaskGroup[],
  maxTasks: number,
): SuggestedTaskGroup[] {
  const limitedGroups: SuggestedTaskGroup[] = [];
  let taskCount = 0;

  for (const group of suggestedTasks) {
    if (taskCount >= maxTasks) break;

    const remainingTasksNeeded = maxTasks - taskCount;
    const tasksToShow = group.tasks.slice(0, remainingTasksNeeded);

    if (tasksToShow.length > 0) {
      limitedGroups.push({
        ...group,
        tasks: tasksToShow,
      });
      taskCount += tasksToShow.length;
    }
  }

  return limitedGroups;
}

export function getDisplayedTaskGroups(
  suggestedTasks: SuggestedTaskGroup[] | undefined,
  isExpanded: boolean,
): SuggestedTaskGroup[] {
  if (!suggestedTasks || suggestedTasks.length === 0) {
    return [];
  }

  if (isExpanded) {
    return suggestedTasks;
  }

  return getLimitedTaskGroups(suggestedTasks, 3);
}

/**
 * Get the repository markdown creation prompt with additional PR creation instructions
 * @param gitProvider The git provider to use for generating provider-specific text
 * @param query Optional custom query to use instead of the default prompt
 * @returns The complete prompt for creating repository markdown and PR instructions
 */
export const getRepoMdCreatePrompt = (
  gitProvider: Provider,
  query?: string,
): string => {
  const providerName = getProviderName(gitProvider);
  const pr = getPR(gitProvider === "gitlab");
  const prShort = getPRShort(gitProvider === "gitlab");

  return `Please explore this repository. Create the file .openhands/microagents/repo.md with:
            ${
              query
                ? `- ${query}`
                : `- A description of the project
            - An overview of the file structure
            - Any information on how to run tests or other relevant commands
            - Any other information that would be helpful to a brand new developer
        Keep it short--just a few paragraphs will do.`
            }

Please push the changes to your branch on ${providerName} and create a ${pr}. Please create a meaningful branch name that describes the changes. If a ${pr} template exists in the repository, please follow it when creating the ${prShort} description.`;
};

/**
 * Get the label for a conversation status
 * @param status The conversation status
 * @returns The localized label for the status
 */
export const getConversationStatusLabel = (
  status: ConversationStatus,
): string => {
  switch (status) {
    case "STOPPED":
      return "COMMON$STOPPED";
    case "RUNNING":
      return "COMMON$RUNNING";
    case "STARTING":
      return "COMMON$STARTING";
    case "ERROR":
      return "COMMON$ERROR";
    case "ARCHIVED":
      return "COMMON$ARCHIVED"; // Use STOPPED for archived conversations
    default:
      return "COMMON$UNKNOWN";
  }
};

// Task Tracking Utility Functions

/**
 * Get the status icon for a task status
 * @param status The task status
 * @returns The emoji icon for the status
 */
export const getStatusIcon = (status: string) => {
  switch (status) {
    case "todo":
      return "⏳";
    case "in_progress":
      return "🔄";
    case "done":
      return "✅";
    default:
      return "❓";
  }
};

/**
 * Get the CSS class names for a task status badge
 * @param status The task status
 * @returns The CSS class names for styling the status badge
 */
export const getStatusClassName = (status: string) => {
  if (status === "done") {
    return "bg-green-800 text-green-200";
  }
  if (status === "in_progress") {
    return "bg-yellow-800 text-yellow-200";
  }
  return "bg-gray-700 text-gray-300";
};

/**
 * Helper function to apply client-side filtering based on search query
 * @param repo The Git repository to check
 * @param searchQuery The search query string
 * @returns True if the repository should be included based on the search query
 */
export const shouldIncludeRepository = (
  repo: GitRepository,
  searchQuery: string,
): boolean => {
  if (!searchQuery.trim()) {
    return true;
  }

  const sanitizedQuery = sanitizeQuery(searchQuery);
  const sanitizedRepoName = sanitizeQuery(repo.full_name);
  return sanitizedRepoName.includes(sanitizedQuery);
};

/**
 * Get the OpenHands query string based on the provider
 * @param provider The git provider
 * @returns The query string for searching OpenHands repositories
 */
export const getOpenHandsQuery = (provider: Provider | null): string => {
  const providerRepositorySuffix: Record<string, string> = {
    gitlab: "openhands-config",
    azure_devops: "openhands-config",
    default: ".openhands",
  } as const;

  return provider && provider in providerRepositorySuffix
    ? providerRepositorySuffix[provider]
    : providerRepositorySuffix.default;
};

/**
 * Check if a repository has the OpenHands suffix based on the provider
 * @param repo The Git repository to check
 * @param provider The git provider
 * @returns True if the repository has the OpenHands suffix
 */
export const hasOpenHandsSuffix = (
  repo: GitRepository,
  provider: Provider | null,
): boolean => repo.full_name.endsWith(`/${getOpenHandsQuery(provider)}`);

/**
 * Build headers for V1 API requests that require session authentication
 * @param sessionApiKey Session API key for authentication
 * @returns Headers object with X-Session-API-Key if provided
 */
export const buildSessionHeaders = (
  sessionApiKey?: string | null,
): Record<string, string> => {
  const headers: Record<string, string> = {};
  if (sessionApiKey) {
    headers["X-Session-API-Key"] = sessionApiKey;
  }
  return headers;
};

/**
 * Check if a task is currently being polled (loading state)
 * @param taskStatus The task status string (e.g., "WORKING", "ERROR", "READY")
 * @returns True if the task is in a loading state (not ERROR and not READY)
 *
 * @example
 * isTaskPolling("WORKING") // Returns true
 * isTaskPolling("PREPARING_REPOSITORY") // Returns true
 * isTaskPolling("READY") // Returns false
 * isTaskPolling("ERROR") // Returns false
 * isTaskPolling(null) // Returns false
 * isTaskPolling(undefined) // Returns false
 */
export const isTaskPolling = (taskStatus: string | null | undefined): boolean =>
  !!taskStatus && taskStatus !== "ERROR" && taskStatus !== "READY";

/**
 * Get the appropriate color based on agent status
 * @param options Configuration object for status color calculation
 * @param options.isPausing Whether the agent is currently pausing
 * @param options.isTask Whether we're polling a task
 * @param options.taskStatus The task status string (e.g., "ERROR", "READY")
 * @param options.isStartingStatus Whether the agent is in a starting state (LOADING or INIT)
 * @param options.isStopStatus Whether the conversation status is STOPPED
 * @param options.curAgentState The current agent state
 * @returns The hex color code for the status
 *
 * @example
 * getStatusColor({
 *   isPausing: false,
 *   isTask: false,
 *   taskStatus: undefined,
 *   isStartingStatus: false,
 *   isStopStatus: false,
 *   curAgentState: AgentState.RUNNING
 * }) // Returns "#BCFF8C"
 */
export const getStatusColor = (options: {
  isPausing: boolean;
  isTask: boolean;
  taskStatus?: string | null;
  isStartingStatus: boolean;
  isStopStatus: boolean;
  curAgentState: AgentState;
}): string => {
  const {
    isPausing,
    isTask,
    taskStatus,
    isStartingStatus,
    isStopStatus,
    curAgentState,
  } = options;

  // Show pausing status
  if (isPausing) {
    return "#FFD600";
  }

  // Show task status if we're polling a task
  if (isTask && taskStatus) {
    if (taskStatus === "ERROR") {
      return "#FF684E";
    }
    return "#FFD600";
  }

  if (isStartingStatus) {
    return "#FFD600";
  }
  if (isStopStatus) {
    return "#ffffff";
  }
  if (curAgentState === AgentState.ERROR) {
    return "#FF684E";
  }
  return "#BCFF8C";
};

interface GetStatusTextArgs {
  isPausing: boolean;
  isTask: boolean;
  taskStatus?: string | null;
  taskDetail?: string | null;
  isStartingStatus: boolean;
  isStopStatus: boolean;
  curAgentState: AgentState;
  errorMessage?: string | null;
  t: (t: string) => string;
}

/**
 * Get the server status text based on agent and task state
 *
 * @param options Configuration object for status text calculation
 * @param options.isPausing Whether the agent is currently pausing
 * @param options.isTask Whether we're polling a task
 * @param options.taskStatus The task status string (e.g., "ERROR", "READY")
 * @param options.taskDetail Optional task-specific detail text
 * @param options.isStartingStatus Whether the conversation is in STARTING state
 * @param options.isStopStatus Whether the conversation is STOPPED
 * @param options.curAgentState The current agent state
 * @param options.errorMessage Optional agent error message
 * @returns Localized human-readable status text
 *
 * @example
 * getStatusText({
 *   isPausing: false,
 *   isTask: true,
 *   taskStatus: "WAITING_FOR_SANDBOX",
 *   taskDetail: null,
 *   isStartingStatus: false,
 *   isStopStatus: false,
 *   curAgentState: AgentState.RUNNING
 * }) // Returns "Waiting For Sandbox"
 */
export function getStatusText({
  isPausing = false,
  isTask,
  taskStatus,
  taskDetail,
  isStartingStatus,
  isStopStatus,
  curAgentState,
  errorMessage,
  t,
}: GetStatusTextArgs): string {
  // Show pausing status
  if (isPausing) {
    return t(I18nKey.COMMON$STOPPING);
  }

  // Show task status if we're polling a task
  if (isTask && taskStatus) {
    if (taskStatus === "ERROR") {
      return taskDetail || t(I18nKey.CONVERSATION$ERROR_STARTING_CONVERSATION);
    }

    if (taskStatus === "READY") {
      return t(I18nKey.CONVERSATION$READY);
    }

    // Format status text: "WAITING_FOR_SANDBOX" -> "Waiting for sandbox"
    return (
      taskDetail ||
      taskStatus
        .toLowerCase()
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    );
  }

  if (isStartingStatus) {
    return t(I18nKey.COMMON$STARTING);
  }

  if (isStopStatus) {
    return t(I18nKey.COMMON$SERVER_STOPPED);
  }

  if (curAgentState === AgentState.ERROR) {
    return errorMessage || t(I18nKey.COMMON$ERROR);
  }

  return t(I18nKey.COMMON$RUNNING);
}
