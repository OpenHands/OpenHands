/**
 * A single feature detected by the critic (e.g., "Insufficient Testing").
 */
export interface CriticFeature {
  /** Internal feature name (e.g., "insufficient_testing") */
  name: string;
  /** Human-readable display name (e.g., "Insufficient Testing") */
  display_name: string;
  /** Probability of this feature being present (0-1) */
  probability: number;
}

/**
 * Categorized features from the critic evaluation.
 */
export interface CriticCategorizedFeatures {
  /** Agent behavioral issues (e.g., insufficient testing, loop behavior) */
  agent_behavioral_issues?: CriticFeature[];
  /** Likely user follow-up patterns */
  user_followup_patterns?: CriticFeature[];
  /** Infrastructure-related issues */
  infrastructure_issues?: CriticFeature[];
  /** Other uncategorized metrics */
  other?: CriticFeature[];
}

/**
 * Metadata from a critic evaluation, including categorized features
 * and event IDs for reproducibility.
 */
export interface CriticMetadata {
  categorized_features?: CriticCategorizedFeatures;
  event_ids?: string[];
  [key: string]: unknown;
}

/**
 * Result of a critic evaluation on an agent's actions.
 *
 * The critic predicts the probability that the agent has successfully
 * completed the task.
 */
export interface CriticResult {
  /** Predicted probability of success (0-1) */
  score: number;
  /** Optional message explaining the score */
  message: string | null;
  /** Optional metadata with categorized features and event IDs */
  metadata: CriticMetadata | null;
}
