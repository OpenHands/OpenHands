import { Provider } from "#/types/settings";

interface GitHubErrorReponse {
  message: string;
  documentation_url: string;
  status: number;
}

interface GitUser {
  id: string;
  login: string;
  avatar_url: string;
  company: string | null;
  name: string | null;
  email: string | null;
}

interface Branch {
  name: string;
  commit_sha: string;
  protected: boolean;
  last_push_date?: string;
}

interface PaginatedBranchesResponse {
  branches: Branch[];
  has_next_page: boolean;
  current_page: number;
  per_page: number;
  total_count?: number;
}

/**
 * V1 API response for paginated branch search (cursor-based)
 */
interface BranchPage {
  items: Branch[];
  next_page_id: string | null;
}

/**
 * V1 API response for paginated repository search (cursor-based)
 */
interface RepositoryPage {
  items: GitRepository[];
  next_page_id: string | null;
}

/**
 * V1 API response for paginated installation search (cursor-based)
 */
interface InstallationPage {
  items: string[];
  next_page_id: string | null;
}

interface GitRepository {
  id: string;
  full_name: string;
  git_provider: Provider;
  is_public: boolean;
  stargazers_count?: number;
  link_header?: string;
  pushed_at?: string;
  owner_type?: "user" | "organization";
  main_branch?: string;
}

interface GitHubCommit {
  html_url: string;
  sha: string;
  commit: {
    author: {
      date: string; // ISO 8601
    };
  };
}

interface GithubAppInstallation {
  installations: { id: number }[];
}
