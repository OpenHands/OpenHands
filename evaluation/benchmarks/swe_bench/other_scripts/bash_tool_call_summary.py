import json
import os
import logging
import matplotlib.pyplot as plt
import argparse
import toml



# qwen_input_file = '/cb/home/harshg/mlf2/agentic_flows/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_100_N_v0.54.0-no-hint-run_1/output.jsonl'
# gpt_oss_file ='/cb/home/harshg/mlf2/agentic_flows/OpenHands/evaluation/evaluation_outputs_aarti_fix_50/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-oss-120b-small_maxiter_100_N_v0.54.0-no-hint-summarizer_for_eval_gptoss-120b-run_1/output.jsonl'
# output_dir_gpt_oss = "/cb/home/harshg/mlf2/agentic_flows/summaries_new/gpt_oss_120b__summary"
# output_dir_qwen = "/cb/home/harshg/mlf2/agentic_flows/summaries_new/qwen_coder_30b_tool_call__summary"

# qwen_480b_input = "/workspaces/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-3-coder-480b_maxiter_250_N_v0.56.0-no-hint-run_1/output.jsonl"
# qwen_480b_output = "/workspaces/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-3-coder-480b_maxiter_250_N_v0.56.0-no-hint-run_1"


# qwen_30b_input = "/workspaces/OpenHands/evaluation/evaluation_outputs_50_cmd_loc/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_250_N_v0.56.0-no-hint-run_1/output.jsonl"
# qwen_30b_output = "/workspaces/OpenHands/evaluation/evaluation_outputs_50_cmd_loc/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/qwen-coder-30b-small_maxiter_250_N_v0.56.0-no-hint-run_1/logs/bash_tools_summary_2"



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
    # Create a directory for plots
    plots_dir = os.path.join(output_dir, "plots", group_name)
    os.makedirs(plots_dir, exist_ok=True)

    # Plot total bash tool calls
    if summary_data['total_bash_tool_calls']:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_data['total_bash_tool_calls'].keys(), summary_data['total_bash_tool_calls'].values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Total Bash Tool Calls for {group_name}")
        plt.xlabel("Bash Tools")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"total_bash_tool_calls.png"))
        plt.close()

    # Plot average bash tool calls per instance
    if summary_data['avg_bash_tool_calls_per_instance']:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_data['avg_bash_tool_calls_per_instance'].keys(), summary_data['avg_bash_tool_calls_per_instance'].values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Average Bash Tool Calls per Instance for {group_name}")
        plt.xlabel("Bash Tools")
        plt.ylabel("Average Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"avg_bash_tool_calls_per_instance.png"))
        plt.close()

    # Plot total tool calls
    if summary_data['total_tool_calls']:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_data['total_tool_calls'].keys(), summary_data['total_tool_calls'].values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Total Tool Calls for {group_name}")
        plt.xlabel("Tools")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"total_tool_calls.png"))
        plt.close()

    # Plot average tool calls per instance
    if summary_data['avg_tool_calls_per_instance']:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(summary_data['avg_tool_calls_per_instance'].keys(), summary_data['avg_tool_calls_per_instance'].values())
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        plt.title(f"Average Tool Calls per Instance for {group_name}")
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
        'avg_tool_calls_per_instance': avg_tool_calls
    }



# Function to save metadata
def save_metadata(group_name, filtered_entries, summary, metadata_dir, logger):
    metadata = {
        "group_name": group_name,
        "entry_count": len(filtered_entries),
        "inclusion_criteria": search_text_groups[group_name].get("include", []),
        "exclusion_criteria": search_text_groups[group_name].get("exclude", []),
        "summary": summary,
        "instance_ids": [entry.get("instance_id") for entry in filtered_entries],
    }
    metadata_file = os.path.join(metadata_dir, f"{group_name}_metadata.json")
    with open(metadata_file, 'w') as meta_outfile:
        json.dump(metadata, meta_outfile, indent=4)
    logger.info(f"Metadata saved to: {metadata_file}")


# Function to filter entries
def filter_entries(input_file, search_text_groups, output_dir, metadata_dir, summary_dir, logger, selected_ids):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    for group_name, criteria in search_text_groups.items():
        include_texts = criteria.get("include", [])
        exclude_texts = criteria.get("exclude", [])
        filtered_entries = []

        for line in lines:
            try:
                entry = json.loads(line)
                text = json.dumps(entry)  # Convert entry back to string for searching

                if selected_ids is not None and entry["instance_id"] not in selected_ids:
                    logger.info(f"Skipping {entry['instance_id']} since not in selected_ids")
                    continue

                # Check inclusion criteria (all must be present)
                if all(inc_text in text for inc_text in include_texts):
                    # Check exclusion criteria (none must be present)
                    if not any(exc_text in line for exc_text in exclude_texts):
                        filtered_entries.append(entry)
            except json.JSONDecodeError:
                logger.info(f"Skipping invalid JSON line: {line}")

        # Create subdirectories for JSON and JSONL files
        json_dir = os.path.join(output_dir, "json")
        jsonl_dir = os.path.join(output_dir, "jsonl")
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(jsonl_dir, exist_ok=True)

        if not filtered_entries:
            print(f"No entries found for group: {group_name}")
            continue
        # Save filtered entries to a new JSONL file
        print(f"Group: {group_name}, Found {len(filtered_entries)} matching entries.")
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

        print(f"---- Evaluating summaries for group: {group_name} {len(filtered_entries)}")
        summary = summary_file(group_name, filtered_entries, summary_dir, logger)

        plot_histograms(summary, group_name, output_dir)


        # Save metadata
        save_metadata(group_name, filtered_entries, summary, metadata_dir, logger)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter JSONL entries based on search criteria.")
    parser.add_argument("--input_file", type=str, help="Path to the input output JSONL file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output files.")
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

    # Additional folder for metadata
    metadata_dir = os.path.join(output_dir, "metadata")
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    search_text_groups = get_search_text_groups()
    logger = setup_logging(output_dir)

    selected_ids = None
    if args.selected_ids is not None:
        selected_ids = toml.load(args.selected_ids)["selected_ids"]

    logger.info(f"Selected_IDS: {selected_ids}")

    # Run the filtering process
    filter_entries(input_file, search_text_groups, output_dir, metadata_dir, summary_dir, logger, selected_ids)
