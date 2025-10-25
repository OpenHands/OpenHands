import os
import docker
import subprocess
from tqdm import tqdm
import time, shutil
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.runtime.builder import DockerRuntimeBuilder


# --- Config ---
client = docker.from_env()
docker_builder = DockerRuntimeBuilder(client)

def instance_id_to_remote_name(instance_id):
    docker_image_prefix = 'docker.io/swebench/'
    repo, name = instance_id.split('__')
    return f'{docker_image_prefix.rstrip("/")}/sweb.eval.x86_64.{repo}_1776_{name}:latest'.lower()

base_image_dir = "/workspaces/Openhands/swebench_dockers_for_eval/swebench_dockers/dev/docker_images"
output_root = "/workspaces/Openhands/swebench_dockers_for_eval/runtime_dockers/swebench_dev"
build_folder = "/workspaces/Openhands/michaelw/build_folder"

tar_files = [f for f in os.listdir(base_image_dir) if f.endswith('.tar')]
# instance_names = [f.removesuffix('__latest.tar') for f in tar_files]
instance_names = ["sqlfluff__sqlfluff-884"]

MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

def is_valid_tar(path: str) -> bool:
    """Quickly verify that a .tar file contains a valid Docker image."""
    if not os.path.exists(path) or os.path.getsize(path) < 1024 * 100:  # <100 KB → suspicious
        return False
    try:
        subprocess.run(
            ["docker", "load", "-i", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False

for instance_name in tqdm(instance_names):
    base_image_name = instance_id_to_remote_name(instance_name)
    output_dir = os.path.join(output_root, instance_name)
    shutil.rmtree(build_folder, ignore_errors=True)
    os.makedirs(build_folder, exist_ok=True)

    # --- Skip if valid tar already exists ---
    if os.path.isdir(output_dir):
        tar_candidates = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".tar")]
        if tar_candidates:
            latest_tar = max(tar_candidates, key=os.path.getmtime)
            if is_valid_tar(latest_tar):
                # print(f"⏭️  Skipping {instance_name} (valid image already exists: {os.path.basename(latest_tar)})")
                continue
            else:
                print(f"⚠️ Found existing tar for {instance_name} but it’s invalid → rebuilding")

    # # --- Build with retries ---
    # for attempt in range(1, MAX_RETRIES + 1):
    #     try:
    #         image_name = build_runtime_image(
    #             base_image_name,
    #             docker_builder,
    #             platform=None,
    #             enable_browser=True,
    #             build_folder=build_folder,
    #         )
    #         print(f"Built image: {image_name}")

    #         os.makedirs(output_dir, exist_ok=True)
    #         safe_image_name = image_name.split("/")[-1].split(":")[-1]
    #         tar_path = os.path.join(output_dir, f"{safe_image_name}.tar")

    #         print(f"Saving image to {tar_path} ...")
    #         subprocess.run(["docker", "save", "-o", tar_path, image_name], check=True)
    #         print(f"✅ Image saved to: {tar_path}")
    #         print(f"🧹 Removing local image {image_name} to free disk space...")
    #         try:
    #             client.images.remove(image=image_name, force=True, noprune=False)
    #             print(f"✅ Removed image: {image_name}")
    #             subprocess.run(
    #                 ["bash", "/workspaces/OpenHands/clean_docker.sh"]
    #             )
    #         except Exception as e:
    #             print(f"⚠️ Could not remove image {image_name}: {e}")

    #         break  # success → exit retry loop

    #     except Exception as e:
    #         print(f"⚠️ Build failed for {instance_name} (attempt {attempt}/{MAX_RETRIES}): {e}")
    #         if attempt < MAX_RETRIES:
    #             print(f"Retrying in {RETRY_DELAY} seconds...")
    #             time.sleep(RETRY_DELAY)
    #         else:
    #             print(f"❌ Giving up on {instance_name} after {MAX_RETRIES} attempts.")
