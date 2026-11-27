import { useTranslation } from "react-i18next";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudMyBudget } from "#/hooks/query/use-cloud-my-budget";
import { I18nKey } from "#/i18n/declaration";

// LiteLLM names the budget that refused the request: a team member is the
// user's own budget in the org, a team is the organization's budget.
const EXCEEDED_BUDGET_PATTERN =
  /Budget has been exceeded! (TeamMember=|User=\S+ in Team=|Team=)/;

function getExceededBudgetScope(
  errorMessage: string | null,
): "user" | "org" | null {
  const match = errorMessage
    ? EXCEEDED_BUDGET_PATTERN.exec(errorMessage)
    : null;
  if (!match) return null;
  return match[1] === "Team=" ? "org" : "user";
}

export interface BudgetLimitError {
  message: string;
  link: { href: string; label: string; openInNewTab: boolean };
}

/**
 * Replaces LiteLLM's raw "Budget has been exceeded" error with a readable one
 * when the active org has budgets on: which budget was reached, its limit,
 * the admin's instructions, and a link to the cloud Your Budget page. Null for
 * any other error and until the budget is known, so the caller keeps the raw
 * message.
 */
export function useBudgetLimitError(
  errorMessage: string | null,
): BudgetLimitError | null {
  const { t, i18n } = useTranslation("openhands");
  const { backend, orgId } = useActiveBackend();
  const scope = getExceededBudgetScope(errorMessage);
  const { data: budget } = useCloudMyBudget({ enabled: scope !== null });

  if (scope === null || !budget?.enabled) return null;
  const limit =
    scope === "user" ? budget.monthly_limit : budget.org_monthly_limit;
  if (typeof limit !== "number") return null;

  const summary = t(
    scope === "user"
      ? I18nKey.ERROR$BUDGET_USER_LIMIT_REACHED
      : I18nKey.ERROR$BUDGET_ORG_LIMIT_REACHED,
    {
      limit: new Intl.NumberFormat(i18n.language, {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: Number.isInteger(limit) ? 0 : 2,
        maximumFractionDigits: 2,
      }).format(limit),
    },
  );
  const instructions = budget.limit_reached_instructions;
  // `org` makes the settings page open on the org that is active here.
  const orgQuery = orgId ? `?org=${encodeURIComponent(orgId)}` : "";

  return {
    message: instructions ? `${summary}\n${instructions}` : summary,
    link: {
      href: `${backend.host.replace(/\/+$/, "")}/settings/your-budget${orgQuery}`,
      label: t(I18nKey.ERROR$BUDGET_VIEW_YOUR_BUDGET),
      // The settings page shares the host when the canvas is locked to it.
      openInNewTab: getLockedCloudHost() === null,
    },
  };
}
