/**
 * Cortex Skills Registry and Interfaces
 * Allows modular extension of agent capabilities with specific expert domains.
 */

export interface CortexSkill {
  id: string;
  name: string;
  description: string;
  version: string;
  tools: string[];
  execute(
    action: string,
    context: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
}

export class CortexSkillsRegistry {
  private skills = new Map<string, CortexSkill>();

  /**
   * Registers a new professional skill to the Cortex platform.
   */
  public register(skill: CortexSkill): void {
    this.skills.set(skill.id, skill);
  }

  /**
   * Retrieves a registered skill by its ID.
   */
  public get(id: string): CortexSkill | undefined {
    return this.skills.get(id);
  }

  /**
   * Lists all registered skills.
   */
  public list(): CortexSkill[] {
    return Array.from(this.skills.values());
  }
}
