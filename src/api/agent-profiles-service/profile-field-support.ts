/**
 * Agent Canvas stores these fields directly on agent profiles in both local and
 * Enterprise cloud backends. Cloud launches now resolve profile-scoped secrets,
 * instructions, tools, skills, and routers from the profile snapshot, while the
 * options endpoint supplies the inherited organization values that the editor
 * can reference.
 */
export function agentProfileSupportsSecretRefs(): boolean {
  return true;
}

export function agentProfileSupportsInstructions(): boolean {
  return true;
}

export function agentProfileSupportsTools(): boolean {
  return true;
}

export function agentProfileSupportsSkills(): boolean {
  return true;
}

export function agentProfileSupportsRouter(): boolean {
  return true;
}
