suggested_task_pr_graphql_query = """
    query GetUserPRs($login: String!) {
        user(login: $login) {
        pullRequests(first: 50, states: [OPEN], orderBy: {field: UPDATED_AT, direction: DESC}) {
            nodes {
            number
            title
            repository {
                nameWithOwner
            }
            mergeable
            commits(last: 1) {
                nodes {
                commit {
                    statusCheckRollup {
                        state
                    }
                }
                }
            }
            reviews(first: 50, states: [CHANGES_REQUESTED, COMMENTED]) {
                nodes {
                state
                }
            }
            }
        }
        }
    }
"""


suggested_task_issue_graphql_query = """
    query GetUserIssues($login: String!) {
        user(login: $login) {
        issues(first: 50, states: [OPEN], filterBy: {assignee: $login}, orderBy: {field: UPDATED_AT, direction: DESC}) {
            nodes {
            number
            title
            repository {
                nameWithOwner
                }
            }
        }
        }
    }
"""

get_thread_from_comment_graphql_query = """
    query GetThreadFromComment($commentId: ID!) {
        node(id: $commentId) {
            ... on PullRequestReviewComment {
                id
                body
                author {
                    login
                }
                createdAt
                updatedAt
                replyTo {
                    id
                    body
                    author {
                        login
                    }
                    createdAt
                    updatedAt
                }
            }
        }
    }
"""

get_review_threads_graphql_query = """
query($owner: String!, $repo: String!, $number: Int!, $first: Int = 50, $after: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      reviewThreads(first: $first, after: $after) {
        nodes {
          id
          path
          isResolved
          comments(first: 1) {
            nodes {
              id
              databaseId
              body
              author {
                login
              }
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

get_thread_comments_graphql_query = """
query ($threadId: ID!, $page: Int = 50, $after: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      id
      path
      isResolved
      comments(first: $page, after: $after) {
        nodes {
          id
          databaseId
          body
          author { login }
          createdAt
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""

# Search branches in a repository by partial name using GitHub GraphQL.
# This leverages the `refs` connection with:
# - refPrefix: "refs/heads/" to restrict to branches
# - query: partial branch name provided by the user
# - first: pagination size (clamped by caller to GitHub limits)
search_branches_graphql_query = """
    query SearchBranches($owner: String!, $name: String!, $query: String!, $perPage: Int!) {
        repository(owner: $owner, name: $name) {
            refs(
                refPrefix: "refs/heads/",
                query: $query,
                first: $perPage,
                orderBy: { field: ALPHABETICAL, direction: ASC }
            ) {
                nodes {
                    name
                    target {
                        __typename
                        ... on Commit {
                            oid
                            committedDate
                        }
                    }
                }
            }
        }
    }
"""
