import { I18nKey } from "#/i18n/declaration";

/**
 * Agent organization: agent profiles arranged like a company. An agent's
 * department is encoded in its profile name as `<department>.<role>` (the
 * profile name rule allows dots), so the arrangement lives on the backend with
 * the profiles themselves and needs no extra storage or API field.
 */

/** Route of the organization chart page (sidebar entry). */
export const ORGANIZATION_PATH = "/organization";

/** Departments in org-chart order. Ids are part of profile names; never rename. */
export type OrganizationDepartmentId =
  | "leadership"
  | "product"
  | "engineering"
  | "quality"
  | "operations";

export interface OrganizationDepartment {
  id: OrganizationDepartmentId;
  titleKey: I18nKey;
  descriptionKey: I18nKey;
}

export const ORGANIZATION_DEPARTMENTS: OrganizationDepartment[] = [
  {
    id: "leadership",
    titleKey: I18nKey.ORG$DEPT_LEADERSHIP,
    descriptionKey: I18nKey.ORG$DEPT_LEADERSHIP_DESC,
  },
  {
    id: "product",
    titleKey: I18nKey.ORG$DEPT_PRODUCT,
    descriptionKey: I18nKey.ORG$DEPT_PRODUCT_DESC,
  },
  {
    id: "engineering",
    titleKey: I18nKey.ORG$DEPT_ENGINEERING,
    descriptionKey: I18nKey.ORG$DEPT_ENGINEERING_DESC,
  },
  {
    id: "quality",
    titleKey: I18nKey.ORG$DEPT_QUALITY,
    descriptionKey: I18nKey.ORG$DEPT_QUALITY_DESC,
  },
  {
    id: "operations",
    titleKey: I18nKey.ORG$DEPT_OPERATIONS,
    descriptionKey: I18nKey.ORG$DEPT_OPERATIONS_DESC,
  },
];

export interface OrganizationRole {
  /** Role segment of the profile name. Never rename. */
  id: string;
  department: OrganizationDepartmentId;
  titleKey: I18nKey;
  descriptionKey: I18nKey;
  /**
   * Appended to the built-in OpenHands system prompt (`system_message_suffix`)
   * so the agent keeps its coding workflow and gains a role. Agent-facing, so
   * it stays in English like the other prompt templates.
   */
  instructions: string;
}

export const ORGANIZATION_ROLES: OrganizationRole[] = [
  {
    id: "tech-lead",
    department: "leadership",
    titleKey: I18nKey.ORG$ROLE_TECH_LEAD,
    descriptionKey: I18nKey.ORG$ROLE_TECH_LEAD_DESC,
    instructions: `You are the Tech Lead of this engineering team.
- Before writing code, restate the goal and break it into small, ordered tasks.
- Choose an approach that fits the existing architecture and explain the trade-offs briefly.
- Delegate focused sub-tasks to sub-agents when it helps, then integrate and check their results.
- Finish with a short summary: what changed, what is left, and any risks.`,
  },
  {
    id: "product-manager",
    department: "product",
    titleKey: I18nKey.ORG$ROLE_PRODUCT_MANAGER,
    descriptionKey: I18nKey.ORG$ROLE_PRODUCT_MANAGER_DESC,
    instructions: `You are the Product Manager.
- Turn each request into a user story, clear requirements, and testable acceptance criteria.
- Ask about anything ambiguous instead of guessing, and list open questions.
- Keep scope small: separate must-haves from nice-to-haves.
- Do not change code unless asked; write specs to markdown files in the workspace when useful.`,
  },
  {
    id: "tech-writer",
    department: "product",
    titleKey: I18nKey.ORG$ROLE_TECH_WRITER,
    descriptionKey: I18nKey.ORG$ROLE_TECH_WRITER_DESC,
    instructions: `You are the Technical Writer.
- Read the code before documenting it; never describe behavior you have not verified.
- Write for the stated audience in short sections with runnable examples.
- Update READMEs, guides, and changelogs to match the current code.
- Keep the existing style and structure of the docs you edit.`,
  },
  {
    id: "backend",
    department: "engineering",
    titleKey: I18nKey.ORG$ROLE_BACKEND,
    descriptionKey: I18nKey.ORG$ROLE_BACKEND_DESC,
    instructions: `You are a Backend Engineer.
- Focus on APIs, data models, persistence, and server-side logic.
- Validate inputs, handle errors explicitly, and keep changes backward compatible unless told otherwise.
- Add or update tests for every behavior change and run them before finishing.
- Follow the project's existing patterns, naming, and dependency choices.`,
  },
  {
    id: "frontend",
    department: "engineering",
    titleKey: I18nKey.ORG$ROLE_FRONTEND,
    descriptionKey: I18nKey.ORG$ROLE_FRONTEND_DESC,
    instructions: `You are a Frontend Engineer.
- Focus on user interfaces: components, state, styling, and accessibility.
- Reuse the project's existing components, design tokens, and i18n setup.
- Make layouts work on small and large screens and keep keyboard and screen-reader support.
- Add or update tests for behavior changes and run them before finishing.`,
  },
  {
    id: "qa",
    department: "quality",
    titleKey: I18nKey.ORG$ROLE_QA,
    descriptionKey: I18nKey.ORG$ROLE_QA_DESC,
    instructions: `You are a QA Engineer.
- Find bugs before users do: list the scenarios, edge cases, and failure modes to cover.
- Write automated tests that follow the project's test conventions, then run them.
- Report each defect with steps to reproduce, expected and actual results.
- Do not change production code unless asked; propose fixes instead.`,
  },
  {
    id: "reviewer",
    department: "quality",
    titleKey: I18nKey.ORG$ROLE_REVIEWER,
    descriptionKey: I18nKey.ORG$ROLE_REVIEWER_DESC,
    instructions: `You are a Code Reviewer.
- Review the current changes (git diff) for correctness, security, performance, and maintainability.
- Rank findings by severity and point to the exact file and line.
- Explain why each issue matters and suggest a concrete fix.
- Do not rewrite the code yourself unless asked.`,
  },
  {
    id: "devops",
    department: "operations",
    titleKey: I18nKey.ORG$ROLE_DEVOPS,
    descriptionKey: I18nKey.ORG$ROLE_DEVOPS_DESC,
    instructions: `You are a DevOps Engineer.
- Focus on CI/CD pipelines, build scripts, containers, and deployment configuration.
- Prefer reproducible, pinned, and least-privilege setups.
- Never print or commit secrets; reference them from the secret store.
- Explain how to verify and roll back every change you make.`,
  },
];

const DEPARTMENT_IDS = new Set<string>(
  ORGANIZATION_DEPARTMENTS.map((department) => department.id),
);

/** Profile name for a role: `<department>.<role>`. */
export function organizationProfileName(role: OrganizationRole): string {
  return `${role.department}.${role.id}`;
}

/**
 * Split a profile name into its department and role. Names without a known
 * department prefix are unassigned (`null`).
 */
export function parseOrganizationProfileName(
  name: string,
): { department: OrganizationDepartmentId; role: string } | null {
  const separator = name.indexOf(".");
  if (separator <= 0 || separator === name.length - 1) return null;
  const department = name.slice(0, separator);
  if (!DEPARTMENT_IDS.has(department)) return null;
  return {
    department: department as OrganizationDepartmentId,
    role: name.slice(separator + 1),
  };
}
