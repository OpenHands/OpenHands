#!/usr/bin/env python3
"""
Python equivalent of the bash script for running SWE-bench evaluations.
Handles Docker image loading, evaluation execution, error logging, and consolidated reporting.
"""

import argparse
import subprocess
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re


class SWEBenchEvaluator:
    def __init__(self, run_id: str, predictions_path: str, output_dir: str = "output", dataset_name: str = "princeton-nlp/SWE-bench", dataset_split="dev"):
        self.run_id = run_id
        self.predictions_path = predictions_path
        self.output_dir = Path(output_dir)
        # self.docker_image_dir = "/mlf3-shared/sapankumars/swe_docker_images"
        self.parent_docker_image_dir = "/workspaces/Openhands/swebench_dockers_for_eval"
        # self.instance_ids_file = "/workspaces/OpenHands/problem_list.txt"
        dockers_dict = {
            "princeton-nlp/SWE-bench_dev": os.path.join(self.parent_docker_image_dir, "swebench_dockers/dev/docker_images"),
            "princeton-nlp/SWE-bench_test": os.path.join(self.parent_docker_image_dir, "swebench_dockers/test/docker_images"),
            "princeton-nlp/SWE-bench_train": os.path.join(self.parent_docker_image_dir, "swebench_dockers/train/docker_images"),
            "princeton-nlp/SWE-bench_Verified_dev": os.path.join(self.parent_docker_image_dir, "swebench_verified_dockers/dev/docker_images"),
        }
        self.docker_image_dir = dockers_dict[f"{dataset_name}_{dataset_split}"]

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        # Results tracking
        self.results = []
        self.consolidated_stats = {
            "total_instances": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "missing_docker_images": 0,
            "successful_evals": 0,
            "failed_evals": 0,
            "skipped_evals": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }

        # SWE-bench evaluation stats (consolidated across all instances)
        self.swe_bench_stats = {
            "total_instances": 0,
            "instances_submitted": 0,
            "instances_completed": 0,
            "instances_incomplete": 0,
            "instances_resolved": 0,
            "instances_unresolved": 0,
            "instances_with_empty_patches": 0,
            "instances_with_errors": 0,
            "unstopped_containers": 0,
            "unremoved_images": 0,
            "completed_ids":[],
            "incomplete_ids":[],
            "resolved_ids":[],
            "unresolved_ids":[],
            "error_ids": [],
            "empty_patch_ids": [],
            "submitted_ids": []
        }


    def setup_logging(self):
        """Setup logging configuration"""
        # Create logs subdirectory in output directory
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Generate timestamp for file naming
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Main log file
        log_file = log_dir / f"swe_bench_eval_{self.run_id}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        # JSONL results file
        self.results_file = log_dir / f"swe_bench_results_{self.run_id}_{timestamp}.jsonl"
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Results file: {self.results_file}")

    def load_instance_ids(self) -> List[str]:
        """Load instance IDs from the specified file"""
        try:
            instance_ids = []
            with open(self.predictions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "instance_id" in data:
                                self.logger.info(f"Found instance_id in predictions: {data['instance_id']}")
                                instance_ids.append(data['instance_id'])
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON line in predictions file: {line.strip()}")


            return instance_ids
        except FileNotFoundError:
            self.logger.error(f"predictions_path file not found: {self.predictions_path}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Error reading predictions_path: {e}")
            sys.exit(1)

    def load_docker_image(self, instance_id: str) -> tuple[bool, str]:
        """
        Load Docker image for the given SWE-bench instance ID.

        Strategy:
        1. Try loading from main tarball directory.
        2. If not found, try fallback directory (e.g., downloaded previously).
        3. If still not found, try pulling from DockerHub with prefetch-style name.
        """
        base_tarfile_path = Path(self.docker_image_dir) / f"{instance_id}__latest.tar"
        fallback_dir = Path(self.docker_image_dir) / "harshg"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_tarfile_path = fallback_dir / f"{instance_id}__latest.tar"

        def try_docker_load(path: Path) -> tuple[bool, str]:
            """Helper to run docker load from a tar file."""
            self.logger.info(f"📦 Attempting to load Docker image from: {path}")
            result = subprocess.run(
                ["docker", "load", "-i", str(path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                self.logger.info(f"✅ Successfully loaded Docker image for {instance_id}")
                return True, "Success"
            else:
                self.logger.warning(f"⚠️ Docker load failed for {instance_id} from {path}: {result.stderr.strip()}")
                return False, result.stderr.strip()

        # --- Step 1: Try loading from main tar folder ---
        if base_tarfile_path.exists():
            ok, msg = try_docker_load(base_tarfile_path)
            if ok:
                return True, msg

        # --- Step 2: Try loading from fallback folder ---
        if fallback_tarfile_path.exists():
            self.logger.info(f"🔁 Checking fallback folder for pre-downloaded image: {fallback_tarfile_path}")
            ok, msg = try_docker_load(fallback_tarfile_path)
            if ok:
                return True, msg
            else:
                self.logger.warning(f"⚠️ Loading from fallback folder failed: {msg}")

        # --- Step 3: Pull from DockerHub (prefetch naming) ---
        repo1 = instance_id.split("__")[0]
        repo2 = instance_id.split("__")[1] if "__" in instance_id else ""
        dockerhub_image = f"swebench/sweb.eval.x86_64.{repo1}_1776_{repo2}:latest"

        self.logger.info(f"🌐 Attempting to pull image from DockerHub: {dockerhub_image}")
        pull_result = subprocess.run(
            ["docker", "pull", dockerhub_image],
            capture_output=True,
            text=True
        )

        if pull_result.returncode == 0:
            self.logger.info(f"✅ Pulled image {dockerhub_image} from DockerHub. Saving locally to fallback...")
            save_result = subprocess.run(
                ["docker", "save", "-o", str(fallback_tarfile_path), dockerhub_image],
                capture_output=True,
                text=True
            )
            if save_result.returncode == 0:
                subprocess.run(["chmod", "777", str(fallback_tarfile_path)], check=False)
                self.logger.info(f"✅ Saved pulled image to {fallback_tarfile_path}")
                subprocess.run(["docker", "rmi", dockerhub_image], capture_output=True, text=True)
                return try_docker_load(fallback_tarfile_path)
            else:
                error_msg = f"❌ Docker save failed: {save_result.stderr.strip()}"
                self.logger.error(error_msg)
                return False, error_msg
        else:
            error_msg = f"❌ Docker pull failed: {pull_result.stderr.strip()}"
            self.logger.error(error_msg)
            return False, error_msg

    def run_evaluation(self, instance_id: str) -> Dict:
        """Run SWE-bench evaluation for the given instance ID"""
        # cmd = [
        #     "python", "-m", "swebench.harness.run_evaluation",
        #     "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        #     "--max_workers", "15",
        #     "--cache_level", "none",
        #     "--clean", "True",
        #     "--predictions_path", self.predictions_path,
        #     "--run_id", self.run_id,
        #     "--instance_ids", instance_id
        # ]

        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", f"{self.dataset_name}",
            "--max_workers", "15",
            "--cache_level", "none",
            "--clean", "True",
            "--predictions_path", self.predictions_path,
            "--run_id", self.run_id,
            "--instance_ids", instance_id,
            "--split", f"{self.dataset_split}"
        ]

        self.logger.info(f"Running evaluation for {instance_id}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Parse evaluation output
            eval_result = self.parse_evaluation_output(instance_id, result.stdout, result.stderr, result.returncode)
            eval_result["instance_id"] = instance_id
            eval_result["command"] = " ".join(cmd)

            if result.returncode == 0:
                self.logger.info(f"Evaluation completed successfully for {instance_id}")
                eval_result["status"] = "success"
            else:
                self.logger.error(f"Evaluation failed for {instance_id}: {result.stderr}")
                eval_result["status"] = "failed"

            return eval_result

        except subprocess.TimeoutExpired:
            self.logger.error(f"Evaluation timed out for {instance_id}")
            return {
                "instance_id": instance_id,
                "status": "timeout",
                "error": "Evaluation timed out after 1 hour"
            }
        except Exception as e:
            self.logger.error(f"Exception during evaluation for {instance_id}: {e}")
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": str(e)
            }

    def parse_evaluation_output(self, instance_id: str, stdout: str, stderr: str, return_code: int) -> Dict:
        """Parse the evaluation output to extract relevant information"""
        result = {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "total_instances": 0,
            "instances_submitted": 0,
            "instances_completed": 0,
            "instances_incomplete": 0,
            "instances_resolved": 0,
            "instances_unresolved": 0,
            "instances_with_empty_patches": 0,
            "instances_with_errors": 0,
            "unstopped_containers": 0,
            "unremoved_images": 0,
            "report_file": None
        }

        # Extract statistics from stdout
        if stdout:
            # Define all the patterns to match
            patterns = {
                "total_instances": r'Total instances: (\d+)',
                "instances_submitted": r'Instances submitted: (\d+)',
                "instances_completed": r'Instances completed: (\d+)',
                "instances_incomplete": r'Instances incomplete: (\d+)',
                "instances_resolved": r'Instances resolved: (\d+)',
                "instances_unresolved": r'Instances unresolved: (\d+)',
                "instances_with_empty_patches": r'Instances with empty patches: (\d+)',
                "instances_with_errors": r'Instances with errors: (\d+)',
                "unstopped_containers": r'Unstopped containers: (\d+)',
                "unremoved_images": r'Unremoved images: (\d+)'
            }

            # Extract values using regex patterns
            for key, pattern in patterns.items():
                match = re.search(pattern, stdout)
                if match:
                    result[key] = int(match.group(1))

                    if key == "instances_completed" and int(match.group(1)) > 0:
                        self.swe_bench_stats["completed_ids"].append(instance_id)
                    elif key == "instances_incomplete" and int(match.group(1)) > 0:
                        self.swe_bench_stats["incomplete_ids"].append(instance_id)
                    elif key == "instances_resolved" and int(match.group(1)) > 0:
                        self.swe_bench_stats["resolved_ids"].append(instance_id)
                    elif key == "instances_unresolved" and int(match.group(1)) > 0:
                        self.swe_bench_stats["unresolved_ids"].append(instance_id)
                    elif key == "instances_with_errors" and int(match.group(1)) > 0:
                        self.swe_bench_stats["error_ids"].append(instance_id)
                    elif key == "instances_with_empty_patches" and int(match.group(1)) > 0:
                        self.swe_bench_stats["empty_patch_ids"].append(instance_id)
                    elif key == "instances_submitted" and int(match.group(1)) > 0:
                        self.swe_bench_stats["submitted_ids"].append(instance_id)

            # Extract report file name
            report_match = re.search(r'Report written to (.+\.json)', stdout)
            if report_match:
                result["report_file"] = report_match.group(1).strip()

        return result

    def log_instance_result(self, instance_id: str, docker_load_success: bool, docker_error: Optional[str], eval_result: Optional[Dict]):
        """Log the result for a single instance to JSONL file"""
        log_entry = {
            "instance_id": instance_id,
            "timestamp": datetime.now().isoformat(),
            "docker_load_success": docker_load_success,
            "docker_load_error": docker_error if not docker_load_success else None,
            "evaluation_result": eval_result
        }

        with open(self.results_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        self.results.append(log_entry)

    def update_consolidated_stats(self, docker_load_success: bool, docker_error: Optional[str], eval_result: Optional[Dict]):
        """Update consolidated statistics"""
        self.consolidated_stats["total_instances"] += 1

        if docker_load_success:
            self.consolidated_stats["successful_loads"] += 1
        else:
            self.consolidated_stats["failed_loads"] += 1
            # Check if it's specifically a missing Docker image
            if docker_error and "Docker image tar not found" in docker_error:
                self.consolidated_stats["missing_docker_images"] += 1

        if eval_result:
            if eval_result.get("status") == "success":
                self.consolidated_stats["successful_evals"] += 1

                # Aggregate SWE-bench specific stats
                swe_stats_keys = [
                    "total_instances", "instances_submitted", "instances_completed",
                    "instances_incomplete", "instances_resolved", "instances_unresolved",
                    "instances_with_empty_patches", "instances_with_errors",
                    "unstopped_containers", "unremoved_images"
                ]

                for key in swe_stats_keys:
                    if key in eval_result:
                        self.swe_bench_stats[key] += eval_result[key]

            else:
                self.consolidated_stats["failed_evals"] += 1
        else:
            # Evaluation was skipped (likely due to Docker load failure)
            self.consolidated_stats["skipped_evals"] += 1

    def process_instance(self, instance_id: str):
        """Process a single instance: load Docker image and run evaluation"""
        self.logger.info(f"Processing task: instance_id={instance_id}")

        # Load Docker image
        docker_load_success, docker_error = self.load_docker_image(instance_id)
        eval_result = None

        if docker_load_success:
            # Run evaluation
            eval_result = self.run_evaluation(instance_id)
        else:
            self.logger.warning(f"Skipping evaluation for {instance_id} due to Docker load failure: {docker_error}")

        # Log results
        self.log_instance_result(instance_id, docker_load_success, docker_error, eval_result)
        self.update_consolidated_stats(docker_load_success, docker_error, eval_result)

    def generate_consolidated_report(self):
        """Generate a consolidated report of all evaluations"""
        self.consolidated_stats["end_time"] = datetime.now().isoformat()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"consolidated_report_{self.run_id}_{timestamp}.json"

        report = {
            "run_id": self.run_id,
            "predictions_path": self.predictions_path,
            "output_directory": str(self.output_dir.absolute()),
            "execution_statistics": self.consolidated_stats,
            "swe_bench_statistics": self.swe_bench_stats,
            "detailed_results": self.results
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Consolidated report written to {report_file}")

        # Print execution summary
        stats = self.consolidated_stats
        self.logger.info("=== EXECUTION SUMMARY ===")
        self.logger.info(f"Total instances processed: {stats['total_instances']}")
        self.logger.info(f"Successful Docker loads: {stats['successful_loads']}")
        self.logger.info(f"Failed Docker loads: {stats['failed_loads']}")
        self.logger.info(f"Missing Docker images: {stats['missing_docker_images']}")
        self.logger.info(f"Successful evaluations: {stats['successful_evals']}")
        self.logger.info(f"Failed evaluations: {stats['failed_evals']}")
        self.logger.info(f"Skipped evaluations: {stats['skipped_evals']}")

        # Print SWE-bench consolidated results
        swe_stats = self.swe_bench_stats
        self.logger.info("=== SWE-BENCH CONSOLIDATED RESULTS ===")
        self.logger.info(f"Total instances: {swe_stats['total_instances']}")
        self.logger.info(f"Instances submitted: {swe_stats['instances_submitted']}")
        self.logger.info(f"Instances completed: {swe_stats['instances_completed']}")
        self.logger.info(f"Instances incomplete: {swe_stats['instances_incomplete']}")
        self.logger.info(f"Instances resolved: {swe_stats['instances_resolved']}")
        self.logger.info(f"Instances unresolved: {swe_stats['instances_unresolved']}")
        self.logger.info(f"Instances with empty patches: {swe_stats['instances_with_empty_patches']}")
        self.logger.info(f"Instances with errors: {swe_stats['instances_with_errors']}")
        self.logger.info(f"Unstopped containers: {swe_stats['unstopped_containers']}")
        self.logger.info(f"Unremoved images: {swe_stats['unremoved_images']}")

        return report_file

    def run(self):
            """Main execution method"""
            self.logger.info(f"Starting SWE-bench evaluation with run_id: {self.run_id}")
            self.logger.info(f"Predictions path: {self.predictions_path}")
            self.logger.info(f"Output directory: {self.output_dir.absolute()}")

            # Load instance IDs
            instance_ids = self.load_instance_ids()

            # Process each instance
            for instance_id in instance_ids:
                try:
                    self.process_instance(instance_id)
                except KeyboardInterrupt:
                    self.logger.info("Evaluation interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error processing {instance_id}: {e}")
                    continue

            # Generate consolidated report
            report_file = self.generate_consolidated_report()
            self.logger.info(f"Evaluation completed! All outputs saved to: {self.output_dir.absolute()}")
            return report_file


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluations")
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument("--predictions_path", required=True, help="Path to predictions file")
    parser.add_argument("--output_dir", default="output", help="Output directory for logs and reports (default: output)")
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench", help="Dataset name for SWE-bench (default: princeton-nlp/SWE-bench)")
    parser.add_argument("--dataset_split", default="dev", help="Dataset name for SWE-bench (default: dev)")


    args = parser.parse_args()

    # Validate arguments
    if not args.run_id:
        parser.error("run_id is required")
    if not args.predictions_path:
        parser.error("predictions_path is required")
    if not Path(args.predictions_path).exists():
        parser.error(f"Predictions path does not exist: {args.predictions_path}")

    # Create and run evaluator
    evaluator = SWEBenchEvaluator(args.run_id, args.predictions_path, args.output_dir, args.dataset_name, args.dataset_split)
    evaluator.run()


if __name__ == "__main__":
    main()