import { useQuery } from "@tanstack/react-query";
import { getCloudOrganizationMe } from "#/api/cloud/organization-service.api";
import { getCloudSetupState } from "#/api/cloud/setup-state-service.api";
import type { CloudSetupGuideSteps } from "#/api/cloud/types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  CONFIG_CACHE_OPTIONS,
  SUPER_ADMIN_SETUP_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import {
  MANAGE_SUPER_ADMINS_PERMISSION,
  SUPER_ADMIN_SETUP_STEPS,
  type SuperAdminSetupStep,
} from "./super-admin-setup-guide.constants";

export interface SuperAdminSetupStepState extends SuperAdminSetupStep {
  done: boolean;
}

const REQUIRED_STEPS = SUPER_ADMIN_SETUP_STEPS.filter(
  (step) => step.completion !== "optional",
);

function isStepDone(
  step: SuperAdminSetupStep,
  guideSteps: CloudSetupGuideSteps | null,
): boolean {
  if (!guideSteps || step.completion === "optional") {
    return false;
  }
  return guideSteps[step.completion];
}

/** The first required step the server does not report as done, if any. */
export function getNextSetupStep(
  guideSteps: CloudSetupGuideSteps | null,
): SuperAdminSetupStep | null {
  return REQUIRED_STEPS.find((step) => !isStepDone(step, guideSteps)) ?? null;
}

/**
 * The enterprise Super Admin setup guide for the active cloud backend.
 *
 * Only instance Super Admins (per `/me` permissions) ask for the setup state,
 * and the server returns a guide only to the first Super Admin while it has
 * an organization and is not dismissed. Local backends never see it.
 */
export function useSuperAdminSetupGuide() {
  const { backend, orgId } = useActiveBackend();
  const isCloud = backend.kind === "cloud";

  // Same query as `useAutomationPermissions`, so no extra request is issued.
  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const meQuery = useQuery({
    queryKey: [
      "cloud-current-user",
      backend.id,
      orgId,
      backend.connectionRevision ?? 0,
    ],
    queryFn: () => getCloudOrganizationMe(orgId!, backend),
    enabled: isCloud && !!orgId,
    staleTime: 1000 * 60 * 5,
    retry: false,
    meta: { disableToast: true },
  });
  const isSuperAdmin =
    meQuery.data?.permissions?.includes(MANAGE_SUPER_ADMINS_PERMISSION) ===
    true;

  // eslint-disable-next-line @tanstack/query/exhaustive-deps
  const setupStateQuery = useQuery({
    queryKey: SUPER_ADMIN_SETUP_QUERY_KEYS.state(
      backend.id,
      backend.connectionRevision ?? 0,
    ),
    queryFn: () => getCloudSetupState(backend),
    enabled: isCloud && isSuperAdmin,
    ...CONFIG_CACHE_OPTIONS,
    retry: false,
    meta: { disableToast: true },
  });

  const setupState = isSuperAdmin ? setupStateQuery.data : undefined;
  const guideSteps = setupState?.guide_steps ?? null;
  const steps: SuperAdminSetupStepState[] = SUPER_ADMIN_SETUP_STEPS.map(
    (step) => ({ ...step, done: isStepDone(step, guideSteps) }),
  );
  const completedCount = REQUIRED_STEPS.filter((step) =>
    isStepDone(step, guideSteps),
  ).length;
  const nextStep = getNextSetupStep(guideSteps);
  const guideOrgId = setupState?.guide_org_id ?? null;
  // Only the first Super Admin's guide has an organization; it stays until dismissed.
  const active = Boolean(guideOrgId) && !setupState?.guide_dismissed;

  return {
    steps,
    completedCount,
    totalCount: REQUIRED_STEPS.length,
    nextStep,
    guideOrgId,
    visible: active && nextStep !== null,
    isLoading: meQuery.isLoading || setupStateQuery.isLoading,
    refetch: setupStateQuery.refetch,
  };
}
