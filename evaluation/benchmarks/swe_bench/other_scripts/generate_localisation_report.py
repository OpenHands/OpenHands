import json
import os
# from transformers import AutoTokenizer
from datasets import load_dataset, disable_progress_bars
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import toml

## disable datasets progress bars
disable_progress_bars()

# ORACLE_DATASET = 'princeton-nlp/SWE-bench_oracle'

# ORACLE_DATASET = 'princeton-nlp/SWE-bench_Verified'

# oracle_ds = load_dataset(ORACLE_DATASET, split='test')

ORACLE_DATASET = 'princeton-nlp/SWE-bench'

oracle_ds = load_dataset(ORACLE_DATASET, split='dev')


## load trajectory
## extract files modified from the patch generated
## extract files needed to be modified from the oracle patch
## compare the two sets of files , and finally output the report into json,


def extract_files_from_patch(patch_text):
    files = set()
    lines = patch_text.split('\n')
    for line in lines:
        if line.startswith('+++ b/'):
            file_path = line[6:]  # Remove '+++ b/' prefix
            files.add(file_path)
    return files


def generate_plots(report, report_dir):


    def gen_plot(data, title, plot_path):
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes.hist(data, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)
        axes.set_title(title)
        axes.set_xlabel('Proportion Correctly Modified')
        axes.set_ylabel('Count')
        axes.set_xlim(0, 1)

        plt.tight_layout()
        plt.show()
        # plt.savefig(f'/mlf-transfers-only/srinjoym/coding_agent_eval/50_problems_eval/pass@1/analysis/localisation_report/mini_swe_agent_{model_name}_localization_accuracy_plots.png', dpi=300, bbox_inches='tight')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')


    def gen_plot_label(data, title, plot_path):
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        n, bins, patches = axes.hist(data, bins=10, range=(0, 1), edgecolor='black', alpha=0.7)

        # Add value labels on top of each bar
        for i in range(len(n)):
            if n[i] > 0:  # Only show labels for non-zero bars
                axes.text(bins[i] + (bins[i+1] - bins[i])/2, n[i] + max(n)*0.01,
                        f'{int(n[i])}', ha='center', va='bottom', fontweight='bold')

        axes.set_title(title)
        axes.set_xlabel('Proportion Correctly Modified')
        axes.set_ylabel('Count')
        axes.set_xlim(0, 1)

        # Adjust y-axis to accommodate labels
        axes.set_ylim(0, max(n) * 1.1)

        plt.tight_layout()
        plt.show()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')


    precision = []
    recall = []


    for i, row_entry in enumerate(report):
        #proportions = [item['proportion_correctly_modified'] for item in data]

        precision.append(row_entry['precision'])
        recall.append(row_entry['recall'])


    precision_plot_path = os.path.join(report_dir, 'localization_precision_plot.png')
    average_precision = sum(precision) / len(precision) if precision else 0
    precision_title = f'localisation_precision_avg_{average_precision:.2f}'

    recall_plot_path = os.path.join(report_dir, 'localization_recall_plot.png')
    average_recall = sum(recall) / len(recall) if recall else 0
    recall_title = f'localisation_recall_avg_{average_recall:.2f}'

    # gen_plot(precision, precision_title, precision_plot_path)
    # gen_plot(recall, recall_title, recall_plot_path)

    gen_plot_label(precision, precision_title, precision_plot_path)
    gen_plot_label(recall, recall_title, recall_plot_path)


def analyse_entry(instance_id, gen_patch):

    oracle_instance = oracle_ds.filter(lambda x: x['instance_id'] == instance_id)[0]
    oracle_patch = oracle_instance['patch']

    print(f" Analysing {instance_id} ---")

    files_modified_by_patch = extract_files_from_patch(gen_patch)
    files_needed_to_be_modified = extract_files_from_patch(oracle_patch)

    ## calculate the intersection over union of these two sets
    intersection = files_modified_by_patch.intersection(files_needed_to_be_modified)
    union = files_modified_by_patch.union(files_needed_to_be_modified)
    iou = len(intersection) / len(union) if union else 0
    recall = len(intersection) /len(files_needed_to_be_modified) if files_needed_to_be_modified else 0
    precision = len(intersection) / len(files_modified_by_patch) if files_modified_by_patch else 0

    unnecessary_edits = []
    edits_correct = []

    for file in list(files_modified_by_patch):
        if file not in files_needed_to_be_modified:
            unnecessary_edits.append(file)
        if file in files_needed_to_be_modified:
            edits_correct.append(file)



    cur_entry = {
        'instance_id': instance_id,
        'files_modified_by_patch': list(files_modified_by_patch),
        'files_needed_to_be_modified': list(files_needed_to_be_modified),
        'proportion_correctly_modified': iou,
        'recall': recall,
        'precision': precision,
        'unnecessary_edits_counts': len(unnecessary_edits),
        'edits_correct_counts': len(edits_correct)
    }


    return cur_entry

def analyse_patches(generated_entries, output_file=None):

    complete_report = []

    for entry in tqdm(generated_entries, desc="Analysing Predictions ..... "):
        instance_id = entry['instance_id']
        gen_patch = entry['model_patch']
        cur_entry_analysis = analyse_entry(instance_id, gen_patch)
        complete_report.append(cur_entry_analysis)

    if output_file is not None:
        with open(output_file, 'w') as f:
            for entry in complete_report:
                f.write(json.dumps(entry) +'\n')

    return complete_report






def main():

    parser = argparse.ArgumentParser(description="Generate localisation report")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model used for generating trajectories')
    parser.add_argument('--predictions_path', type=str, required=True, help='Path to the predictions JSONL file')
    parser.add_argument('--report_dir', type=str, required=True, help='Directory to save the localisation report and plots')
    parser.add_argument('--selected_ids', type=str, required=False, default=None, help="Pass toml file with key selected_ids")
    args = parser.parse_args()

    model_name = args.model_name
    predictions_path = args.predictions_path
    localisation_report_dir = args.report_dir

    os.makedirs(localisation_report_dir, exist_ok=True)

    selected_ids = None
    if args.selected_ids is not None:
        selected_ids = toml.load(args.selected_ids)["selected_ids"]

    print(f"Selected_IDS: {selected_ids}")

    predictions_data = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_line = json.loads(line)
            instance_id = data_line["instance_id"]
            if selected_ids is not None:
                if instance_id in selected_ids:
                    predictions_data.append(data_line)
            else:
                predictions_data.append(data_line)


    localisation_report_path = os.path.join(localisation_report_dir, f'{model_name}_localisation_report.jsonl')

    print('Reading predictions from file:', predictions_path)
    print('Model used:', model_name)
    print('report saved at:', localisation_report_path)

    complete_report = []

    print(f'Len of predictions_data:',{len(predictions_data)})
    complete_report = analyse_patches(predictions_data, localisation_report_path)

    print('Generating plots...')

    generate_plots(complete_report, localisation_report_dir)

if __name__ == '__main__':
    main()
