import json
import os
import logging
import matplotlib.pyplot as plt
import argparse
import toml



def get_search_text_groups():
    search_text_groups = {
        "Empty_patches": {
            "include": ['"test_result": {"git_patch": ""}'],
            "exclude": []
        },
        "Non-empty_patches": {
            "include": ['"test_result": {"git_patch": "diff'],
            "exclude": []
        },
        "agentGotStuckError": {
            "include": ['AgentStuckInLoopError: Agent got stuck in a loop'],
            "exclude": []
        },
        "agentGotStuckError_and_empty_patches": {
            "include": ['AgentStuckInLoopError: Agent got stuck in a loop','"test_result": {"git_patch": ""}'],
            "exclude": []
        },
        "agentGotStuckError_and_non_empty_patches": {
            "include": ['AgentStuckInLoopError: Agent got stuck in a loop','"test_result": {"git_patch": "diff'],
            "exclude": []
        },
        "without_errror_and_non_empty_patches": {
            "include": ['"test_result": {"git_patch": "diff'],
            "exclude": ["AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED_ERROR", "AgentStuckInLoopError: Agent got stuck in a loop",'RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100','RuntimeError: There was an unexpected error while','STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR']
        },
        "without_error_and_empty_patches": {
            "include": ['"test_result": {"git_patch": ""}'],
            "exclude": ["AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED_ERROR", "AgentStuckInLoopError: Agent got stuck in a loop",'RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100','RuntimeError: There was an unexpected error while','STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR']
        },
        'without_error_patches': {
            'include': [],
            'exclude': ["AGENT_ERROR$ERROR_ACTION_NOT_EXECUTED_ERROR", "AgentStuckInLoopError: Agent got stuck in a loop",'RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100','RuntimeError: There was an unexpected error while','STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR']
        },
        'max_iters_error_and_non_empty_patches' : {
            "include": ['RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100','"test_result": {"git_patch":"diff'],
            "exclude": []
        },
        "max_iters_error_and_empty_patches" : {
            "include": ['RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100','"test_result": {"git_patch": ""}'],
            "exclude": []
        },
        'max_iters_error' : {
            "include": ['RuntimeError: Agent reached maximum iteration. Current iteration: 100, max iteration: 100'],
            "exclude": []
        },
        "runtime_errors":{
            "include": ['RuntimeError: There was an unexpected error while'],
            "exclude": []
        },
        "runtime_error_empty_patches":{
            "include": ['RuntimeError: There was an unexpected error while','"test_result": {"git_patch": ""}'],
            "exclude": []
        },
        "runtime_error_non_empty_patches":{
            "include": ['RuntimeError: There was an unexpected error while','"test_result": {"git_patch": "diff'],
            "exclude": []
        },
        "llm_internal_server_error":{
            "include": ['STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR'],
            "exclude": []
        },
    }
    return search_text_groups



def plot_histograms(summary_data, group_name, output_dir):

    def sort_data(data_dict):
        return dict(sorted(data_dict.items(), key=lambda item: item[1], reverse=True))
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, "plots", group_name)
    os.makedirs(plots_dir, exist_ok=True)

    num_instances = summary_data['num_instances']

    # Plot total bash tool calls
    if summary_data['total_bash_tool_calls']:
        sorted_data = sort_data(summary_data['total_bash_tool_calls'])
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_data.keys(), sorted_data.values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Total Bash Tool Calls for {group_name}={sum(list(sorted_data.values()))}, num_instances={num_instances}")
        plt.xlabel("Bash Tools")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"total_bash_tool_calls.png"))
        plt.close()

    # Plot average bash tool calls per instance
    if summary_data['avg_bash_tool_calls_per_instance']:
        sorted_data = sort_data(summary_data['avg_bash_tool_calls_per_instance'])
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_data.keys(), sorted_data.values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Average Bash Tool Calls per Instance for {group_name}, num_instances={num_instances}")
        plt.xlabel("Bash Tools")
        plt.ylabel("Average Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"avg_bash_tool_calls_per_instance.png"))
        plt.close()

    # Plot total tool calls
    if summary_data['total_tool_calls']:
        sorted_data = sort_data(summary_data['total_tool_calls'])
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_data.keys(), sorted_data.values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Total Tool Calls for {group_name}={sum(list(sorted_data.values()))}, num_instances={num_instances}")
        plt.xlabel("Tools")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"total_tool_calls.png"))
        plt.close()

    # Plot average tool calls per instance
    if summary_data['avg_tool_calls_per_instance']:
        sorted_data = sort_data(summary_data['avg_tool_calls_per_instance'])
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_data.keys(), sorted_data.values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Average Tool Calls per Instance for {group_name}, num_instances={num_instances}")
        plt.xlabel("Tools")
        plt.ylabel("Average Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"avg_tool_calls_per_instance.png"))
        plt.close()



def add_item(dict_obj, key, value):
    if key in dict_obj:
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [value]


def eval_entry(entry, logger):
    history = entry['history'][3:]
    conversation_size = len(history)
    events = []
    out_source = []
    bash_tool_summary = {}  # {'ls': {'count': , list: [{'id', 'command' }]}, {}
    source = {}
    source_ids_list = {}
    events_list = []
    tool_call_counts = {}
    agent_messages = []
    user_messages = {}
    observations = {}
    agent_calls = []
    # observation_list = [ {id,action, output}]
    # action_list = [ {id, action, command} ]


    for item in history:


        ## Count source occurrences and store ids
        if item['source'] not in source_ids_list:
            source_ids_list[item['source']] = []
            source[item['source']] = 0
        source_ids_list[item['source']].append(item['id'])
        source[item['source']] +=1


        action = item.get('action', '')
        observation = item.get('observation', '')
        try:
            if observation!= '':
                events_list.append(f"id_{item['id']}_{item['source']}_observation_{observation} {item['message']}")
                add_item(observations, item['source'], str({'id': item['id'], 'source': item['source'] ,'observation': observation ,'message': item['message']}))
            elif action != '':

                if item['source'] == 'agent':
                    ### tools to summarize  "edit/run/finish/message/think/"
                    if action in ['edit', 'run', 'finish', 'think', 'task_tracking', 'run_ipython', 'view', 'read']:
                        function_name = item.get('tool_call_metadata', {}).get('function_name', '')
                        events_list.append(f"id_{item['id']}_{item['source']}_action_{action}_{function_name} {item['message']}")
                        agent_calls.append({'id': item['id'], 'action': action, 'function_name': item['tool_call_metadata']['function_name'], 'message': item['message']})
                        ### tool call summary #####
                        tool_name = item['tool_call_metadata']['function_name']
                        if tool_name not in tool_call_counts:
                            tool_call_counts[tool_name] = 0
                        tool_call_counts[tool_name] += 1


                        #### bash tool summary #####
                        if action == 'run':
                            if item['tool_call_metadata']['function_name'] == 'execute_bash':
                                command = item['message'].split('Running command: ')[-1].strip()
                                for cmd in command.split('&&'):
                                    prefix = cmd.strip().split(' ')[0]
                                    cmd = cmd.strip()
                                    prefix = prefix.split('<')[0].split('>')[0].split('=')[0]  # Handle redirection
                                    if prefix not in bash_tool_summary:
                                        bash_tool_summary[prefix] = {'count': 0, 'list': []}
                                    bash_tool_summary[prefix]['count'] += 1
                                    bash_tool_summary[prefix]['list'].append({'id': item['id'], 'command': command})
                        ###########################

                    ##### empty messages #####
                    elif action == 'message':
                        events_list.append(f"id_{item['id']}_{item['source']}_action_{action} {item['message']}")
                        if item['message'].strip() == '':
                            agent_messages.append({'id': item['id'], 'message': item['message']})
                    ###########################
                    elif action == 'condensation':
                        events_list.append(f"id_{item['id']}_{item['source']}_action_{action} {item['message']}")
                        agent_calls.append({'id': item['id'], 'action': action, 'function_name':'condensation','message': item['message']})
                    else:
                        logger.warning(f"Unrecognized agent action without tool_call_metadata: {item}")

                elif item['source'] == 'user':
                    events_list.append(f"id_{item['id']}_{item['source']}_action_{action} {item['message']}")
                    add_item(user_messages, action, {'id': item['id'], 'message': item['message']})

                else:
                    logger.warning(f"Unrecognized source with action: {item}")


            else:
                logger.warning(f"\n\nItem with no action or observation: {item}\n\n")
        except Exception as e:
            logger.error(f"\n\nError processing item {item}: {e}\n\n")
    bash_tool_call_counts = {tool: details['count'] for tool, details in bash_tool_summary.items()}

    summary = {
        'instance_id': entry.get('instance_id', ''),
        'error': entry.get('error', ''),
        'conversation_size': conversation_size,
        'source_counts': source,
        'tool_call_counts': tool_call_counts,
        'bash_tool_call_counts': bash_tool_call_counts,
        'agent_calls': agent_calls,
        'agent_messages': agent_messages,
        'observations': observations,
        'user_messages': user_messages,
        'events': events_list,
        'bash_tool_call_summary': bash_tool_summary,
        'source_ids': source_ids_list,
    }

    return summary


def summary_file(group_name, filtered_entries, summary_dir, logger):
    summaries = []
    bash_tool_calls = {}
    tool_calls = {}
    for entry in filtered_entries:
        summary = eval_entry(entry, logger)
        summaries.append(summary)
        # Aggregate bash tool calls
        for tool, count in summary['bash_tool_call_counts'].items():
            if tool not in bash_tool_calls:
                bash_tool_calls[tool] = 0
            bash_tool_calls[tool] += count

        # tool_call_counts
        for tool, count in summary['tool_call_counts'].items():
            if tool not in tool_calls:
                tool_calls[tool] = 0
            tool_calls[tool] += count

        ### avg tool calls per instance
    avg_tool_calls = {tool: count / len(filtered_entries) for tool, count in tool_calls.items()}
    avg_bash_tool_calls = {tool: count / len(filtered_entries) for tool, count in bash_tool_calls.items()}

    # Save summaries to a JSON file
    summary_file_path = os.path.join(summary_dir, f"{group_name}_summary.json")
    with open(summary_file_path, 'w') as summary_outfile:
        json.dump(summaries, summary_outfile, indent=4)
    logger.info(f"Summary saved to: {summary_file_path}")

    return {
        'total_bash_tool_calls': bash_tool_calls,
        'avg_bash_tool_calls_per_instance': avg_bash_tool_calls,
        'total_tool_calls': tool_calls,
        'avg_tool_calls_per_instance': avg_tool_calls,
        'num_instances': len(filtered_entries)
    }



# Function to save metadata
def save_metadata(group_name, filtered_entries, summary, metadata_dir, logger, overall_stats):
    metadata = {
        "group_name": group_name,
        "entry_count": len(filtered_entries),
        "inclusion_criteria": search_text_groups[group_name].get("include", []),
        "exclusion_criteria": search_text_groups[group_name].get("exclude", []),
        "summary": summary,
        "instance_ids": [entry.get("instance_id") for entry in filtered_entries],
        "overall_instances_in_this_category": overall_stats
    }
    metadata_file = os.path.join(metadata_dir, f"{group_name}_metadata.json")
    with open(metadata_file, 'w') as meta_outfile:
        json.dump(metadata, meta_outfile, indent=4)
    logger.info(f"Metadata saved to: {metadata_file}")


def _filter_instances(loc_file, result_file, logger, selected_ids):
    with open(loc_file, 'r') as loc_fh:
        loc_data = [json.loads(line) for line in loc_fh if line.strip()]

        if selected_ids is not None:
            loc_data = [x for x in loc_data if x["instance_id"] in selected_ids]
            logger.info(f"selected_data: {len(loc_data)}")


    recall_eq_1 = [item['instance_id'] for item in loc_data if item.get('recall', 0) == 1.0]
    recall_less_1 = [item['instance_id'] for item in loc_data if item.get('recall', 0) < 1.0]

    logger.info(f"Instances with recall=1: {len(recall_eq_1)}/{len(loc_data)}")
    logger.info(f"Instances with recall < 1: {len(recall_less_1)}/{len(loc_data)}")

    with open(result_file, 'r') as res_fh:
        result_data = json.load(res_fh)

    stats = result_data["swe_bench_statistics"]
    resolved = stats["resolved_ids"]
    if selected_ids is not None:
        resolved = [x for x in resolved if x in selected_ids]
    unresolved = stats["unresolved_ids"]
    if selected_ids is not None:
        unresolved = [x for x in unresolved if x in selected_ids]

    total_instances = len(loc_data)

    logger.info(f"Resolved instances: {len(resolved)}/{total_instances}")
    logger.info(f"Unresolved instances: {len(unresolved)}/{total_instances}")

    all_instances = [x["instance_id"] for x in loc_data]

    return recall_eq_1, recall_less_1, resolved, unresolved, all_instances


# Function to filter entries
def filter_entries(input_file, search_text_groups, output_dir, metadata_dir, summary_dir, logger, filtered_instances=None):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    overall_stats = {"total_entries": 0, "all_instance_ids": []}
    for line in lines:
        entry = json.loads(line)
        if filtered_instances is not None:
            if entry["instance_id"] in filtered_instances:
                overall_stats["total_entries"] += 1
                overall_stats["all_instance_ids"].append(entry["instance_id"])
        else:
            overall_stats["total_entries"] += 1
            overall_stats["all_instance_ids"].append(entry["instance_id"])

    for group_name, criteria in search_text_groups.items():
        logger.info(f"------------- Processing group: {group_name} -------------")
        include_texts = criteria.get("include", [])
        exclude_texts = criteria.get("exclude", [])
        filtered_entries = []

        for line in lines:
            try:
                entry = json.loads(line)
                if filtered_instances is not None and entry["instance_id"] not in filtered_instances:
                    logger.info(f"Skipping instance_id {entry['instance_id']} not in filtered instances.")
                    continue
                text = json.dumps(entry)  # Convert entry back to string for searching

                # if group_name == "without_errror_and_non_empty_patches":
                #     logger.info(f"Checking instance_id {entry['instance_id']}")
                #     logger.info(f"Entry line: {line}")
                #     logger.info(f"Entry text: {text}")
                #     logger.info(f"Include texts: {include_texts}")
                #     logger.info(f"Exclude texts: {exclude_texts}")
                #     logger.info(f"Include check: {all(inc_text in line for inc_text in include_texts)}")
                #     logger.info(f"Exclude check: {any(exc_text in line for exc_text in exclude_texts)}")

                #     logger.info(f"Include check: {all(inc_text in text for inc_text in include_texts)}")
                #     logger.info(f"Exclude check: {any(exc_text in text for exc_text in exclude_texts)}")

                # Check inclusion criteria (all must be present)
                if all(inc_text in text for inc_text in include_texts):
                    # Check exclusion criteria (none must be present)
                    if not any(exc_text in text for exc_text in exclude_texts):
                        filtered_entries.append(entry)
            except json.JSONDecodeError:
                logger.info(f"Skipping invalid JSON line: {text}")

        # Create subdirectories for JSON and JSONL files
        json_dir = os.path.join(output_dir, "json")
        jsonl_dir = os.path.join(output_dir, "jsonl")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(jsonl_dir, exist_ok=True)

        if not filtered_entries:
            logger.info(f"No entries found for group: {group_name}")
            continue
        # Save filtered entries to a new JSONL file
        logger.info(f"Group: {group_name}, Found {len(filtered_entries)} matching entries.")
        jsonl_output_file = os.path.join(jsonl_dir, f"{group_name}_filtered.jsonl")
        with open(jsonl_output_file, 'w') as outfile:
            for entry in filtered_entries:
                outfile.write(json.dumps(entry) + '\n')
        logger.info(f"Filtered entries saved to: {jsonl_output_file}")

        # Save filtered entries to a new JSON file
        json_output_file = os.path.join(json_dir, f"{group_name}_filtered.json")
        with open(json_output_file, 'w') as json_outfile:
            json.dump(filtered_entries, json_outfile, indent=4)
        logger.info(f"Filtered entries saved to: {json_output_file}")


        summary = summary_file(group_name, filtered_entries, summary_dir, logger)

        plot_histograms(summary, group_name, output_dir)


        # Save metadata
        save_metadata(group_name, filtered_entries, summary, metadata_dir, logger, overall_stats)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter JSONL entries based on search criteria.")
    parser.add_argument("--input_file", required=True, type=str, help="Path to the input JSONL file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save the output files.")
    parser.add_argument("--loc_json", type=str, help="Path to the localization JSON file with instance details.", default=None, required=True)
    parser.add_argument("--eval_json", type=str, help="Path to the eval_summary JSON file with instance details.", default=None, required=True)
    parser.add_argument('--selected_ids', type=str, required=False, default=None, help="Pass toml file with key selected_ids")

    return parser.parse_args()


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



if __name__ == "__main__":

    args = parse_arguments()
    input_file = args.input_file
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    search_text_groups = get_search_text_groups()
    logger = setup_logging(output_dir)

    selected_ids = None
    if args.selected_ids is not None:
        selected_ids = toml.load(args.selected_ids)["selected_ids"]

    logger.info(f"Selected_IDS: {selected_ids}")

    recall_eq_1, recall_less_1, resolved, unresolved, all_instances = _filter_instances(args.loc_json, args.eval_json, logger, selected_ids)


    # Run the filtering process - recall=1
    if recall_eq_1:
        output_recall_1 = os.path.join(output_dir, "recall_eq_1")
        os.makedirs(output_recall_1, exist_ok=True)
        metadata_dir = os.path.join(output_recall_1, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_recall_1, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        filter_entries(input_file, search_text_groups, output_recall_1, metadata_dir, summary_dir, logger, recall_eq_1)

    # Run the filtering process - recall < 1
    if recall_less_1:
        output_recall_less_1 = os.path.join(output_dir, "recall_less_1")
        os.makedirs(output_recall_less_1, exist_ok=True)
        metadata_dir = os.path.join(output_recall_less_1, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_recall_less_1, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        filter_entries(input_file, search_text_groups, output_recall_less_1, metadata_dir, summary_dir, logger, recall_less_1)

    # Run the filtering process - resolved
    if resolved:
        output_resolved = os.path.join(output_dir, "resolved")
        os.makedirs(output_resolved, exist_ok=True)
        metadata_dir = os.path.join(output_resolved, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_resolved, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        filter_entries(input_file, search_text_groups, output_resolved, metadata_dir, summary_dir, logger, resolved)


    # Run the filtering process - unresolved
    if unresolved:
        output_unresolved = os.path.join(output_dir, "unresolved")
        os.makedirs(output_unresolved, exist_ok=True)
        metadata_dir = os.path.join(output_unresolved, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_unresolved, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        filter_entries(input_file, search_text_groups, output_unresolved, metadata_dir, summary_dir, logger, unresolved)

    # Run the filtering process - recall_1 and resolved
    recall_1_and_resolved = list(set(recall_eq_1) & set(resolved))
    if recall_1_and_resolved:
        output_recall_1_and_resolved = os.path.join(output_dir, "recall_eq_1_and_resolved")
        os.makedirs(output_recall_1_and_resolved, exist_ok=True)
        metadata_dir = os.path.join(output_recall_1_and_resolved, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_recall_1_and_resolved, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        filter_entries(input_file, search_text_groups, output_recall_1_and_resolved, metadata_dir, summary_dir, logger, recall_1_and_resolved)

    # Run the filtering process - recall_1 and unresolved
    recall_1_and_unresolved = list(set(recall_eq_1) & set(unresolved))
    if recall_1_and_unresolved:
        output_recall_1_and_unresolved = os.path.join(output_dir, "recall_eq_1_and_unresolved")
        os.makedirs(output_recall_1_and_unresolved, exist_ok=True)
        metadata_dir = os.path.join(output_recall_1_and_unresolved, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        summary_dir = os.path.join(output_recall_1_and_unresolved, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        filter_entries(input_file, search_text_groups, output_recall_1_and_unresolved, metadata_dir, summary_dir, logger, recall_1_and_unresolved)

    # Run the filtering process - no filter
    output_no_filter = os.path.join(output_dir, "no_filter")
    os.makedirs(output_no_filter, exist_ok=True)
    metadata_dir = os.path.join(output_no_filter, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    summary_dir = os.path.join(output_no_filter, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    filter_entries(input_file, search_text_groups, output_no_filter, metadata_dir, summary_dir, logger, filtered_instances=all_instances)

    logger.info("Processing completed.")
