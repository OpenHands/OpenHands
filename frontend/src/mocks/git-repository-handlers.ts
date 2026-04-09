import { delay, http, HttpResponse } from "msw";
import { GitRepository, Branch, PaginatedBranchesResponse } from "#/types/git";
import { Provider } from "#/types/settings";

// Generate a list of mock repositories with realistic data
const generateMockRepositories = (
  count: number,
  provider: Provider,
): GitRepository[] =>
  Array.from({ length: count }, (_, i) => ({
    id: `${i + 1}`,
    full_name: `user/repo-${i + 1}`,
    git_provider: provider,
    is_public: Math.random() > 0.3, // 70% chance of being public
    stargazers_count: Math.floor(Math.random() * 1000),
    pushed_at: new Date(
      Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000,
    ).toISOString(), // Last 90 days
    owner_type: Math.random() > 0.7 ? "organization" : "user", // 30% chance of being organization
  }));

// Generate mock branches for a repository
const generateMockBranches = (count: number): Branch[] =>
  Array.from({ length: count }, (_, i) => ({
    name: (() => {
      if (i === 0) return "main";
      if (i === 1) return "develop";
      return `feature/branch-${i}`;
    })(),
    commit_sha: `abc123${i.toString().padStart(3, "0")}`,
    protected: i === 0, // main branch is protected
    last_push_date: new Date(
      Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000,
    ).toISOString(),
  }));

// Mock repositories for each provider
const MOCK_REPOSITORIES = {
  github: generateMockRepositories(120, "github"),
  gitlab: generateMockRepositories(120, "gitlab"),
  bitbucket: generateMockRepositories(120, "bitbucket"),
};

// Mock branches (same for all repos for simplicity)
const MOCK_BRANCHES = generateMockBranches(25);

export const GIT_REPOSITORY_HANDLERS = [
  http.get("/api/user/repositories", async ({ request }) => {
    await delay(500); // Simulate network delay

    const url = new URL(request.url);
    const selectedProvider = url.searchParams.get("selected_provider");
    const page = parseInt(url.searchParams.get("page") || "1", 10);
    const perPage = parseInt(url.searchParams.get("per_page") || "30", 10);
    const sort = url.searchParams.get("sort") || "pushed";
    const installationId = url.searchParams.get("installation_id");

    // Simulate authentication error if no provider token
    if (!selectedProvider) {
      return HttpResponse.json(
        "Git provider token required. (such as GitHub).",
        { status: 401 },
      );
    }

    // Get repositories for the selected provider
    const repositories =
      MOCK_REPOSITORIES[selectedProvider as keyof typeof MOCK_REPOSITORIES] ||
      [];

    // Sort repositories based on the sort parameter
    let sortedRepos = [...repositories];
    if (sort === "pushed") {
      sortedRepos.sort(
        (a, b) =>
          new Date(b.pushed_at!).getTime() - new Date(a.pushed_at!).getTime(),
      );
    } else if (sort === "stars") {
      sortedRepos.sort(
        (a, b) => (b.stargazers_count || 0) - (a.stargazers_count || 0),
      );
    }

    // Handle installation filtering (for GitHub Apps)
    if (installationId && selectedProvider === "github") {
      // Simulate filtering by installation - in real API this would filter by access
      const installationIndex = parseInt(installationId, 10) || 0;
      const startRepo = installationIndex * 20; // Each installation has ~20 repos
      sortedRepos = sortedRepos.slice(startRepo, startRepo + 20);
    }

    // Calculate pagination
    const startIndex = (page - 1) * perPage;
    const endIndex = startIndex + perPage;
    const paginatedRepos = sortedRepos.slice(startIndex, endIndex);
    const hasNextPage = endIndex < sortedRepos.length;
    const hasPrevPage = page > 1;
    const totalPages = Math.ceil(sortedRepos.length / perPage);

    // Generate GitHub-style link header for pagination
    let linkHeader = "";
    if (hasNextPage || hasPrevPage) {
      const links = [];
      if (hasPrevPage) {
        links.push(
          `</api/user/repositories?page=${page - 1}&per_page=${perPage}>; rel="prev"`,
        );
      }
      if (hasNextPage) {
        links.push(
          `</api/user/repositories?page=${page + 1}&per_page=${perPage}>; rel="next"`,
        );
      }
      links.push(
        `</api/user/repositories?page=${totalPages}&per_page=${perPage}>; rel="last"`,
      );
      links.push(
        `</api/user/repositories?page=1&per_page=${perPage}>; rel="first"`,
      );
      linkHeader = links.join(", ");
    }

    // Add link_header to the first repository if pagination info exists
    const responseRepos = [...paginatedRepos];
    if (responseRepos.length > 0 && linkHeader) {
      responseRepos[0] = { ...responseRepos[0], link_header: linkHeader };
    }

    // Return response as direct Repository array (matching real API)
    return HttpResponse.json(responseRepos);
  }),

  http.get("/api/user/search/repositories", async ({ request }) => {
    await delay(300); // Simulate network delay

    const url = new URL(request.url);
    const query = url.searchParams.get("query") || "";
    const selectedProvider = url.searchParams.get("selected_provider");
    const perPage = parseInt(url.searchParams.get("per_page") || "5", 10);
    const sort = url.searchParams.get("sort") || "stars";
    const order = url.searchParams.get("order") || "desc";

    // Simulate authentication error if no provider token
    if (!selectedProvider) {
      return HttpResponse.json("Git provider token required.", {
        status: 401,
      });
    }

    // Get repositories for the selected provider
    const repositories =
      MOCK_REPOSITORIES[selectedProvider as keyof typeof MOCK_REPOSITORIES] ||
      [];

    // Filter repositories by search query
    const filteredRepos = repositories.filter((repo) =>
      repo.full_name.toLowerCase().includes(query.toLowerCase()),
    );

    // Sort repositories
    const sortedRepos = [...filteredRepos];
    if (sort === "stars") {
      sortedRepos.sort((a, b) => {
        const aStars = a.stargazers_count || 0;
        const bStars = b.stargazers_count || 0;
        return order === "desc" ? bStars - aStars : aStars - bStars;
      });
    }

    // Limit results
    const limitedRepos = sortedRepos.slice(0, perPage);

    return HttpResponse.json(limitedRepos);
  }),

  // Repository branches endpoint
  http.get("/api/user/repository/branches", async ({ request }) => {
    await delay(300);

    const url = new URL(request.url);
    const repository = url.searchParams.get("repository");
    const page = parseInt(url.searchParams.get("page") || "1", 10);
    const perPage = parseInt(url.searchParams.get("per_page") || "30", 10);

    if (!repository) {
      return HttpResponse.json("Repository parameter is required", {
        status: 400,
      });
    }

    // Calculate pagination
    const startIndex = (page - 1) * perPage;
    const endIndex = startIndex + perPage;
    const paginatedBranches = MOCK_BRANCHES.slice(startIndex, endIndex);
    const hasNextPage = endIndex < MOCK_BRANCHES.length;

    const response: PaginatedBranchesResponse = {
      branches: paginatedBranches,
      has_next_page: hasNextPage,
      current_page: page,
      per_page: perPage,
      total_count: MOCK_BRANCHES.length,
    };

    return HttpResponse.json(response);
  }),

  // Search repository branches endpoint
  http.get("/api/user/search/branches", async ({ request }) => {
    await delay(200);

    const url = new URL(request.url);
    const repository = url.searchParams.get("repository");
    const query = url.searchParams.get("query") || "";
    const perPage = parseInt(url.searchParams.get("per_page") || "30", 10);

    if (!repository) {
      return HttpResponse.json("Repository parameter is required", {
        status: 400,
      });
    }

    // Filter branches by search query
    const filteredBranches = MOCK_BRANCHES.filter((branch) =>
      branch.name.toLowerCase().includes(query.toLowerCase()),
    );

    // Limit results
    const limitedBranches = filteredBranches.slice(0, perPage);

    return HttpResponse.json(limitedBranches);
  }),
];
