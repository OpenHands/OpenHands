import json
import logging
import argparse
import os
import toml
from bash_tool_call_summary_filtered import get_search_text_groups
from bash_tool_call_summary_filtered import eval_entry


def get_groups_matched(entry, search_text_groups, logger):
    matched_groups = []
    text = json.dumps(entry)
    for group_name, criteria in search_text_groups.items():

        logger.info(f"------------- Processing group: {group_name} -------------")
        include_texts = criteria.get("include", [])
        exclude_texts = criteria.get("exclude", [])

        # Check inclusion criteria (all must be present)
        if all(inc_text in text for inc_text in include_texts):
            # Check exclusion criteria (none must be present)
            if not any(exc_text in text for exc_text in exclude_texts):
                matched_groups.append(group_name)

    return matched_groups


def single_instance_summary(entry, localization_report, eval_report, search_text_groups, logger):
    summary_instance = {
        'instance_id': entry['instance_id'],
        'error': entry.get('error', ''),
        'conversation_size': None,
        'source_counts': None,
        'tool_call_counts': None,
        'bash_tool_call_counts': None,
        'matched_groups': None,
        'is_resolved': False,
        'precision': None,
        'recall': None,
        'files_modified_by_patch': None,
        'files_needed_to_be_modified': None,
        'llm_metrics': {
            "num_calls_to_llm": 0,
            'prompt_tokens': None,
            'completion_tokens': None
        }
    }

    summary = eval_entry(entry, logger)
    matched_groups = get_groups_matched(entry, search_text_groups, logger)
    summary_instance['matched_groups'] = matched_groups

    for k, val in summary.items():
        if k in summary_instance:
            summary_instance[k] = val

    found_last = False
    for item in reversed(entry['history']):
        if 'llm_metrics' in item:
            if item['source'] == 'agent':
                summary_instance['llm_metrics']['num_calls_to_llm'] += 1
                if not found_last:
                    summary_instance['llm_metrics']['prompt_tokens'] = item['llm_metrics']['accumulated_token_usage']['prompt_tokens']
                    summary_instance['llm_metrics']['completion_tokens'] = item['llm_metrics']['accumulated_token_usage']['completion_tokens']
                    found_last = True

    for line in localization_report:
        if line['instance_id'] == entry['instance_id']:
            summary_instance['precision'] = line['precision']
            summary_instance['recall'] = line['recall']
            summary_instance['files_modified_by_patch'] = line['files_modified_by_patch']
            summary_instance['files_needed_to_be_modified'] = line['files_needed_to_be_modified']

    swe_bench_statistics = eval_report['swe_bench_statistics']
    if entry['instance_id'] in swe_bench_statistics['resolved_ids']:
        summary_instance['is_resolved'] = True

    return summary_instance


def build_summary_of_run(input_file, eval_summary_file, localization_report, selected_ids, logger):

    search_text_groups = get_search_text_groups()

    with open(eval_summary_file, 'r') as eval_fh:
        eval_report_data = json.load(eval_fh)

    loc_report_data = []
    with open(localization_report, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            loc_report_data.append(json.loads(line))

    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    overall_summary = []
    for line in lines:
        try:
            entry = json.loads(line)
            if selected_ids is not None and entry["instance_id"] not in selected_ids:
                logger.info(f"Skipping instance_id {entry['instance_id']} not in selected_ids instances.")
                continue

            summary_instance = single_instance_summary(entry, loc_report_data, eval_report_data, search_text_groups, logger)
            overall_summary.append(summary_instance)

        except json.JSONDecodeError:
            logger.info(f"Skipping invalid JSON line: {line}")

    fname = f"overall_summary.json" if selected_ids is None else f"overall_summary_selected.json"
    output_file = os.path.join(os.path.dirname(input_file), fname)
    with open(output_file, 'w') as outfile:
        json.dump(overall_summary, outfile, indent=4)

    logger.info(f"Summary saved to {output_file}")


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "summary_openhands.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter JSONL entries based on search criteria.")
    parser.add_argument("--input_file", required=True, type=str, help="Path to the input JSONL file.")
    parser.add_argument("--eval_summary_file", required=True, type=str, help="Path to the evaluation summary JSON file.")
    parser.add_argument("--localization_report", required=True, type=str, help="Path to the localization report JSONL file.")
    parser.add_argument('--selected_ids', type=str, required=False, default=None, help="Path to TOML file with key 'selected_ids'")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    input_file = args.input_file
    eval_summary_file = args.eval_summary_file
    localization_report = args.localization_report
    output_dir = os.path.dirname(input_file)

    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    selected_ids = None
    if args.selected_ids is not None:
        selected_ids = toml.load(args.selected_ids)["selected_ids"]

    logger.info(f"Selected_IDS: {selected_ids}")

    build_summary_of_run(input_file, eval_summary_file, localization_report, selected_ids, logger)
