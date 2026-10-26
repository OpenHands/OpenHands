import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { Provider } from "#/types/settings";
import { AgentState } from "#/types/agent-state";
import { I18nKey } from "#/i18n/declaration";
import { getTaskStatusI18nKey } from "#/utils/status";
import type { AppConversationStartTaskStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";

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
 * Detect a phone/tablet user agent (Android, iPhone, iPad, …).
 * Unlike isMobileDevice, this ignores touch capability, so a desktop OS with
 * a touchscreen (e.g. a Windows 2-in-1) is NOT matched. Use this when the
 * decision depends on having a physical keyboard rather than on touch input.
 */
export const isMobileUserAgent = (): boolean =>
  /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent,
  );

/**
 * Detect an Apple platform (macOS, iOS, iPadOS), whose keyboard shortcuts use
 * ⌘ where other platforms use Ctrl. iPadOS Safari reports a Mac user agent,
 * which this matches too.
 */
export const isApplePlatform = (): boolean =>
  typeof navigator !== "undefined" &&
  /Macintosh|Mac OS X|iPhone|iPad|iPod/.test(navigator.userAgent);

/**
 * Detect if the user is on a mobile device.
 * Touch support alone is not sufficient — touchscreen laptops have touch
 * but use a mouse/trackpad as primary input. We check that the primary
 * pointing device is coarse (finger) to avoid false positives.
 */
export const isMobileDevice = (): boolean => {
  if (isMobileUserAgent()) return true;

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
 * Whether to use the installation-scoped repo flow
 * (`/api/v1/git/installations/search` → `/api/v1/git/repositories/search?installation_id=…`)
 * for the given provider/backend combo.
 *
 * Mirrors OpenHands' cloud frontend (parameterized by `app_mode`):
 *   - bitbucket / bitbucket_data_center → always installation-based
 *   - github / azure_devops → installation-based ONLY when the active backend
 *     is cloud (Azure DevOps organizations are its installations)
 *   - gitlab / forgejo → direct (search) flow
 *
 * `appMode` accepts the active backend `kind` ("local" | "cloud") so call
 * sites can hand it through directly.
 */
export const shouldUseInstallationRepos = (
  provider: Provider | null | undefined,
  appMode?: "local" | "cloud",
) => {
  if (!provider) return false;

  switch (provider) {
    case "bitbucket":
    case "bitbucket_data_center":
      return true;
    case "github":
    case "azure_devops":
      return appMode === "cloud";
    default:
      return false;
  }
};

export const getGitProviderBaseUrl = (
  gitProvider: Provider,
  host?: string | null,
): string => {
  // If custom host provided, use it (with https:// prefix if needed)
  if (host && host.trim() !== "") {
    return host.startsWith("http") ? host : `https://${host}`;
  }

  // Fall back to defaults
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
  if (gitProvider === "bitbucket_data_center") return "Bitbucket Data Center";
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
 * constructPullRequestUrl(789, "bitbucket", "PROJECT/repo", "server.com") // "https://server.com/projects/PROJECT/repos/repo/pull-requests/789"
 */
export const constructPullRequestUrl = (
  prNumber: number,
  provider: Provider,
  repositoryName: string,
  host?: string | null,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider, host);

  switch (provider) {
    case "github":
      return `${baseUrl}/${repositoryName}/pull/${prNumber}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/pull/${prNumber}`;
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/merge_requests/${prNumber}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/pull-requests/${prNumber}`;
    case "bitbucket_data_center": {
      const [project, repo] = repositoryName.split("/");
      return `${baseUrl}/projects/${project}/repos/${repo}/pull-requests/${prNumber}`;
    }
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
 * Percent-encode one URL path segment. Names parsed from a remote URL arrive
 * encoded (`Web%20Apps`) while provider APIs return them raw (`Web Apps`), so
 * decode first to avoid encoding twice.
 */
const encodePathSegment = (segment: string): string => {
  try {
    return encodeURIComponent(decodeURIComponent(segment));
  } catch {
    return encodeURIComponent(segment);
  }
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
 * constructRepositoryUrl("azure_devops", "org/Web Apps/repo") // "https://dev.azure.com/org/Web%20Apps/_git/repo"
 */
export const constructRepositoryUrl = (
  provider: Provider,
  repositoryName: string,
  host?: string | null,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider, host);
  if (provider === "bitbucket_data_center") {
    const [project, repo] = repositoryName.split("/");
    return `${baseUrl}/projects/${project}/repos/${repo}`;
  }
  if (provider === "azure_devops") {
    // Azure DevOps format: org/project/repo
    const parts = repositoryName.split("/");
    if (parts.length === 3) {
      const [org, project, repo] = parts.map(encodePathSegment);
      return `${baseUrl}/${org}/${project}/_git/${repo}`;
    }
  }
  return `${baseUrl}/${repositoryName}`;
};

/**
 * Percent-encode a branch name for use as a URL path component. Each
 * `/`-separated segment is encoded on its own so nested branch names such as
 * `release/1.0` keep their slashes while `#`, `%` and `&` are escaped.
 */
const encodeBranchPath = (branchName: string): string =>
  branchName.split("/").map(encodeURIComponent).join("/");

/**
 * Construct the branch URL for different providers
 * @param provider The git provider
 * @param repositoryName The repository name in format "owner/repo"
 * @param branchName The branch name
 * @param host Optional custom host for self-hosted instances
 * @returns The branch URL
 *
 * @example
 * constructBranchUrl("github", "owner/repo", "main") // "https://github.com/owner/repo/tree/main"
 * constructBranchUrl("gitlab", "owner/repo", "develop") // "https://gitlab.com/owner/repo/-/tree/develop"
 * constructBranchUrl("bitbucket", "owner/repo", "feature") // "https://bitbucket.org/owner/repo/src/feature"
 * constructBranchUrl("bitbucket", "PROJECT/repo", "feature", "server.com") // "https://server.com/projects/PROJECT/repos/repo/browse?at=refs/heads/feature"
 */
export const constructBranchUrl = (
  provider: Provider,
  repositoryName: string,
  branchName: string,
  host?: string | null,
): string => {
  const baseUrl = getGitProviderBaseUrl(provider, host);

  switch (provider) {
    case "github":
      return `${baseUrl}/${repositoryName}/tree/${encodeBranchPath(branchName)}`;
    case "forgejo":
      return `${baseUrl}/${repositoryName}/src/branch/${encodeBranchPath(branchName)}`;
    case "gitlab":
      return `${baseUrl}/${repositoryName}/-/tree/${encodeBranchPath(branchName)}`;
    case "bitbucket":
      return `${baseUrl}/${repositoryName}/src/${encodeBranchPath(branchName)}`;
    case "bitbucket_data_center": {
      // Bitbucket Server format: /projects/{PROJECT}/repos/{repo}/browse?at=refs/heads/{branch}
      const parts = repositoryName.split("/");
      if (parts.length >= 2) {
        const [project, repo] = parts;
        // The branch is one query value here, so encode it whole: an
        // unencoded `&` would start a new query parameter.
        return `${baseUrl}/projects/${project}/repos/${repo}/browse?at=refs/heads/${encodeURIComponent(branchName)}`;
      }
      return "";
    }
    case "azure_devops": {
      // Azure DevOps format: org/project/repo
      const parts = repositoryName.split("/");
      if (parts.length === 3) {
        const [org, project, repo] = parts;
        // The branch is one query value here, so encode it whole: an
        // unencoded `&` would start a new query parameter.
        return `${baseUrl}/${org}/${project}/_git/${repo}?version=GB${encodeURIComponent(branchName)}`;
      }
      return "";
    }
    default:
      return "";
  }
};

// Git Action Prompts

const GIT_COMMIT_PROMPT =
  "Please review the current changes and create a git commit with a concise, descriptive message.";

export const getGitCommitPrompt = (): string => GIT_COMMIT_PROMPT;

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
 * Generate a create new branch prompt
 * @returns The create new branch prompt
 */
export const getCreateNewBranchPrompt = (): string =>
  "Please create a new branch with a descriptive name related to the work you plan to do.";

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
 * }) // Returns "var(--oh-status-success)"
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
      return OH_STATUS_ERROR_COLOR;
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
    return OH_STATUS_ERROR_COLOR;
  }
  return OH_STATUS_SUCCESS_COLOR;
};

interface GetStatusTextArgs {
  isPausing: boolean;
  isTask: boolean;
  taskStatus?: AppConversationStartTaskStatus | null;
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
 *   taskStatus: "STARTING_CONVERSATION",
 *   taskDetail: null,
 *   isStartingStatus: false,
 *   isStopStatus: false,
 *   curAgentState: AgentState.RUNNING
 * }) // Returns "Starting conversation"
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

    return taskDetail || t(getTaskStatusI18nKey(taskStatus));
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
