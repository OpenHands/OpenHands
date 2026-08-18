import { createHash } from 'crypto';

export const GENESIS_HASH = '0000000000000000000000000000000000000000000000000000000000000000';

export interface DebtAuditReceipt {
  index: number;
  timestamp: string;
  repoPath: string;
  eventType: string;
  readinessScore: number;
  criticalSmells: string[];
  prevHash: string;
  currHash: string;
  metadata: Record<string, unknown>;
}

export interface TechnicalDebtReport {
  repoPath: string;
  readinessScore: number;
  isProductionReady: boolean;
  debtBreakdown: {
    loopDebtScore: number;
    tokenInflationScore: number;
    mutationSafetyScore: number;
    cyclomaticDebtScore: number;
  };
  criticalSmells: string[];
  auditReceipt: DebtAuditReceipt;
}

export class ActionGateDueDiligenceLedger {
  private entries: DebtAuditReceipt[] = [];
  private lastHash: string = GENESIS_HASH;

  public recordAudit(
    repoPath: string,
    eventType: string,
    readinessScore: number,
    criticalSmells: string[],
    metadata: Record<string, unknown> = {}
  ): DebtAuditReceipt {
    const timestamp = new Date().toISOString();
    const index = this.entries.length;

    const metaString = JSON.stringify(metadata);
    const metaHash = createHash('sha256').update(metaString).digest('hex');

    const canonical = `${index}|${this.lastHash}|${repoPath}|${eventType}|${readinessScore}|${timestamp}|${metaHash}`;
    const currHash = createHash('sha256').update(canonical).digest('hex');

    const receipt: DebtAuditReceipt = {
      index,
      timestamp,
      repoPath,
      eventType,
      readinessScore,
      criticalSmells,
      prevHash: this.lastHash,
      currHash,
      metadata,
    };

    this.entries.push(receipt);
    this.lastHash = currHash;
    return receipt;
  }

  public getEntries(): DebtAuditReceipt[] {
    return [...this.entries];
  }

  public verifyIntegrity(): boolean {
    let prev = GENESIS_HASH;
    for (const entry of this.entries) {
      if (entry.prevHash !== prev) {
        return false;
      }
      prev = entry.currHash;
    }
    return true;
  }
}

export class TechnicalDebtAnalyzer {
  private ledger: ActionGateDueDiligenceLedger;
  public readonly neverEquateIntentToApproval: boolean;
  public readonly minimumReadinessThreshold: number;

  constructor(
    minimumReadinessThreshold: number = 75,
    neverEquateIntentToApproval: boolean = true
  ) {
    this.ledger = new ActionGateDueDiligenceLedger();
    this.minimumReadinessThreshold = minimumReadinessThreshold;
    this.neverEquateIntentToApproval = neverEquateIntentToApproval;
  }

  public getLedger(): ActionGateDueDiligenceLedger {
    return this.ledger;
  }

  public checkKillSwitch(): boolean {
    const envVal = (process.env.AAG_KILL_SWITCH || '').toLowerCase();
    return envVal === 'true' || envVal === '1' || envVal === 'yes';
  }

  public analyzeCodebase(
    repoPath: string,
    metrics: {
      unboundedLoops?: number;
      tokenMultiplier?: number;
      unGatedMutations?: number;
      cyclomaticComplexityAverage?: number;
    } = {}
  ): TechnicalDebtReport {
    if (this.checkKillSwitch()) {
      const receipt = this.ledger.recordAudit(
        repoPath,
        'audit_halted_kill_switch',
        0,
        ['EMERGENCY_KILL_SWITCH_ACTIVE'],
        { reason: 'AAG_KILL_SWITCH is set' }
      );
      throw new Error('A2Z SOC ActionGate: Emergency kill switch is engaged. Technical due diligence halted.');
    }

    const unboundedLoops = metrics.unboundedLoops || 0;
    const tokenMultiplier = metrics.tokenMultiplier || 1.0;
    const unGatedMutations = metrics.unGatedMutations || 0;
    const cyclomaticAvg = metrics.cyclomaticComplexityAverage || 5.0;

    const criticalSmells: string[] = [];

    // 1. Loop Debt
    let loopDebtScore = 100 - unboundedLoops * 25;
    if (loopDebtScore < 0) loopDebtScore = 0;
    if (unboundedLoops > 0) {
      criticalSmells.push(`DETECTED_${unboundedLoops}_UNBOUNDED_REASONING_LOOPS`);
    }

    // 2. Token Inflation Debt
    let tokenInflationScore = Math.max(0, 100 - (tokenMultiplier - 1.0) * 50);
    if (tokenMultiplier > 2.0) {
      criticalSmells.push('HIGH_TOKEN_INFLATION_CASCADE');
    }

    // 3. Mutation Safety Debt (Zero-Trust ActionBoundary)
    let mutationSafetyScore = 100 - unGatedMutations * 30;
    if (mutationSafetyScore < 0) mutationSafetyScore = 0;
    if (unGatedMutations > 0) {
      criticalSmells.push(`DETECTED_${unGatedMutations}_UNGATED_PRODUCTION_MUTATIONS`);
    }

    // 4. Cyclomatic & Complexity Debt
    let cyclomaticDebtScore = Math.max(0, 100 - (cyclomaticAvg - 5) * 5);

    // Aggregate Score
    const rawScore =
      loopDebtScore * 0.3 +
      tokenInflationScore * 0.25 +
      mutationSafetyScore * 0.3 +
      cyclomaticDebtScore * 0.15;
    const readinessScore = Math.round(rawScore * 10) / 10;
    const isProductionReady = readinessScore >= this.minimumReadinessThreshold && criticalSmells.length === 0;

    const auditReceipt = this.ledger.recordAudit(
      repoPath,
      isProductionReady ? 'diligence_passed' : 'diligence_failed_production_debt',
      readinessScore,
      criticalSmells,
      {
        metrics,
        neverEquateIntentToApproval: this.neverEquateIntentToApproval,
      }
    );

    return {
      repoPath,
      readinessScore,
      isProductionReady,
      debtBreakdown: {
        loopDebtScore,
        tokenInflationScore,
        mutationSafetyScore,
        cyclomaticDebtScore,
      },
      criticalSmells,
      auditReceipt,
    };
  }
}
