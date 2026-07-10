export const BUDGET_TABS = [
  { value: "organization", label: "Organization budget" },
  { value: "defaults", label: "Default budgets" },
  { value: "overrides", label: "User overrides" },
] as const;

export const USERS_PER_PAGE = 50;

export type BudgetTab = (typeof BUDGET_TABS)[number]["value"];
