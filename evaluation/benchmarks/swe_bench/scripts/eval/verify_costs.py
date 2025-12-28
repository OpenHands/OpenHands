import argparse

import pandas as pd

from openhands.core.logger import openhands_logger as logger


def verify_instance_costs(row: pd.Series) -> float:
    """Verifies that the accumulated_cost matches the sum of individual costs in metrics.
    Also checks for duplicate consecutive costs which might indicate buggy counting.
    If the consecutive costs are identical, the file is affected by this bug:
    https://github.com/OpenHands/OpenHands/issues/5383

    Args:
        row: DataFrame row containing instance data with metrics
    Returns:
        float: The verified total cost for this instance (corrected if needed)
    """
    try:
        metrics = row.get('metrics')
        if not metrics:
            logger.warning(f'Instance {row["instance_id"]}: No metrics found')
            return 0.0

        accumulated = metrics.get('accumulated_cost')
        costs = metrics.get('costs', [])

        if accumulated is None:
            logger.warning(
                f'Instance {row["instance_id"]}: No accumulated_cost in metrics'
            )
            return 0.0

        # Check for duplicate consecutive costs and systematic even-odd pairs
        has_duplicate = False
        all_pairs_match = True

        # Check each even-odd pair (0-1, 2-3, etc.)
        for i in range(0, len(costs) - 1, 2):
            if abs(costs[i]['cost'] - costs[i + 1]['cost']) < 1e-6:
                has_duplicate = True
                logger.debug(
                    f'Instance {row["instance_id"]}: Possible buggy double-counting detected! '
                    f'Steps {i} and {i + 1} have identical costs: {costs[i]["cost"]:.2f}'
                )
            else:
                all_pairs_match = False
                break

        # Calculate total cost, accounting for buggy double counting if detected
        if len(costs) >= 2 and has_duplicate and all_pairs_match:
            paired_steps_cost = sum(
                cost_entry['cost']
                for cost_entry in costs[: -1 if len(costs) % 2 else None]
            )
            real_paired_cost = paired_steps_cost / 2

            unpaired_cost = costs[-1]['cost'] if len(costs) % 2 else 0
            total_cost = real_paired_cost + unpaired_cost

        else:
            total_cost = sum(cost_entry['cost'] for cost_entry in costs)

        if not abs(total_cost - accumulated) < 1e-6:
            logger.warning(
                f'Instance {row["instance_id"]}: Cost mismatch: '
                f'accumulated: {accumulated:.2f}, sum of costs: {total_cost:.2f}, '
            )

        return total_cost

    except Exception as e:
        logger.error(
            f'Error verifying costs for instance {row.get("instance_id", "UNKNOWN")}: {e}'
        )
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Verify costs in SWE-bench output file'
    )
    parser.add_argument(
        'input_filepath', type=str, help='Path to the output.jsonl file'
    )
    args = parser.parse_args()

    try:
        # Load and verify the JSONL file
        df = pd.read_json(args.input_filepath, lines=True)
        logger.info(f'Loaded {len(df)} instances from {args.input_filepath}')

        # Verify costs for each instance and sum up total
        total_cost = df.apply(verify_instance_costs, axis=1).sum()
        logger.info(f'Total verified cost across all instances: ${total_cost:.2f}')

    except Exception as e:
        logger.error(f'Failed to process file: {e}')
        raise


if __name__ == '__main__':
    main()
