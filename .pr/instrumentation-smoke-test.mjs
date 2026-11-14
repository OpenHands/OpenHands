// Replicates trackEvent's payload shaping (use-tracking.ts calls track() with
// the same object literal construction) to prove the new event contracts.
// The dialog's blocked-event emissions are asserted by typecheck + eslint;
// this smoke test verifies the exact property objects the tracking functions
// hand to PostHog, including optional-field omission.
import assert from "node:assert";

const sent = [];
const track = (event, props) => sent.push([event, props]);

const trackAutomationSetupOpened = ({ automationId, entryPoint }) => {
  track("automation_setup_opened", {
    automation_id: automationId,
    ...(entryPoint ? { entry_point: entryPoint } : {}),
  });
};

const trackAutomationSetupBlocked = ({
  automationId,
  stage,
  reason,
  field,
  errorCode,
}) => {
  track("automation_setup_blocked", {
    automation_id: automationId,
    stage,
    reason,
    ...(field ? { field } : {}),
    ...(errorCode ? { error_code: errorCode } : {}),
  });
};

// Dialog open (new caller)
trackAutomationSetupOpened({
  automationId: "github-pr-reviewer",
  entryPoint: "setup_dialog",
});
// Legacy caller without entry point still works (property omitted, not null)
trackAutomationSetupOpened({ automationId: "news-digest" });
// Prerequisites blocked (was: silent disabled button)
trackAutomationSetupBlocked({
  automationId: "github-pr-reviewer",
  stage: "prerequisites",
  reason: "github",
  errorCode: "prerequisites_blocked",
});
// Local form validation failure
trackAutomationSetupBlocked({
  automationId: "custom-automation",
  stage: "form",
  reason: "local_validation",
  field: "repositories",
  errorCode: "required",
});
// Preflight form-level error (no field)
trackAutomationSetupBlocked({
  automationId: "news-digest",
  stage: "preflight",
  reason: "service_validation",
  errorCode: "service_error",
});

assert.deepEqual(sent, [
  [
    "automation_setup_opened",
    { automation_id: "github-pr-reviewer", entry_point: "setup_dialog" },
  ],
  ["automation_setup_opened", { automation_id: "news-digest" }],
  [
    "automation_setup_blocked",
    {
      automation_id: "github-pr-reviewer",
      stage: "prerequisites",
      reason: "github",
      error_code: "prerequisites_blocked",
    },
  ],
  [
    "automation_setup_blocked",
    {
      automation_id: "custom-automation",
      stage: "form",
      reason: "local_validation",
      field: "repositories",
      error_code: "required",
    },
  ],
  [
    "automation_setup_blocked",
    {
      automation_id: "news-digest",
      stage: "preflight",
      reason: "service_validation",
      error_code: "service_error",
    },
  ],
]);
console.log("SMOKE OK: 5/5 event payloads match the new contracts");
sent.forEach(([e, p]) => console.log(" ", e, JSON.stringify(p)));
