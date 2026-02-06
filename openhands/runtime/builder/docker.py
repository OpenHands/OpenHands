# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import datetime
import os
import subprocess
import time

import docker

from openhands.core.exceptions import AgentRuntimeBuildError
from openhands.core.logger import RollingLogger
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.builder.base import RuntimeBuilder
from openhands.utils.term_color import TermColor, colorize
from openhands.version import get_version


class DockerRuntimeBuilder(RuntimeBuilder):
    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client

        version_info = self.docker_client.version()
        server_version = version_info.get('Version', '').replace('-', '.')
        components = version_info.get('Components')
        self.is_podman = (
            components is not None
            and len(components) > 0
            and components[0].get('Name', '').startswith('Podman')
        )
        if (
            tuple(map(int, server_version.split('.')[:2])) < (18, 9)
            and not self.is_podman
        ):
            raise AgentRuntimeBuildError(
                'Docker server version must be >= 18.09 to use BuildKit'
            )

        if self.is_podman and tuple(map(int, server_version.split('.')[:2])) < (4, 9):
            raise AgentRuntimeBuildError('Podman server version must be >= 4.9.0')

        self.rolling_logger = RollingLogger(max_lines=10)

    @staticmethod
    def check_buildx(is_podman: bool = False) -> bool:
        """Check if Docker Buildx is available."""
        try:
            result = subprocess.run(
                ['docker' if not is_podman else 'podman', 'buildx', 'version'],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def build(
        self,
        path: str,
        tags: list[str],
        platform: str | None = None,
        extra_build_args: list[str] | None = None,
        use_local_cache: bool = False,
    ) -> str:
        """Builds a Docker image using BuildKit and handles the build logs appropriately.

        Args:
            path (str): The path to the Docker build context.
            tags (list[str]): A list of image tags to apply to the built image.
            platform (str, optional): The target platform for the build. Defaults to None.
            use_local_cache (bool, optional): Whether to use and update the local build cache. Defaults to True.
            extra_build_args (list[str], optional): Additional arguments to pass to the Docker build command. Defaults to None.

        Returns:
            str: The name of the built Docker image.

        Raises:
            AgentRuntimeBuildError: If the Docker server version is incompatible or if the build process fails.

        Note:
            This method uses Docker BuildKit for improved build performance and caching capabilities.
            If `use_local_cache` is True, it will attempt to use and update the build cache in a local directory.
            The `extra_build_args` parameter allows for passing additional Docker build arguments as needed.
        """
        self.docker_client = docker.from_env()
        version_info = self.docker_client.version()
        server_version = version_info.get('Version', '').split('+')[0].replace('-', '.')
        components = version_info.get('Components')
        self.is_podman = (
            components is not None
            and len(components) > 0
            and components[0].get('Name', '').startswith('Podman')
        )
        if tuple(map(int, server_version.split('.'))) < (18, 9) and not self.is_podman:
            raise AgentRuntimeBuildError(
                'Docker server version must be >= 18.09 to use BuildKit'
            )

        if self.is_podman and tuple(map(int, server_version.split('.'))) < (4, 9):
            raise AgentRuntimeBuildError('Podman server version must be >= 4.9.0')

        if not DockerRuntimeBuilder.check_buildx(self.is_podman):
            # when running openhands in a container, there might not be a "docker"
            # binary available, in which case we need to download docker binary.
            # since the official openhands app image is built from debian, we use
            # debian way to install docker binary
            logger.info(
                'No docker binary available inside openhands-app container, trying to download online...'
            )
            commands = [
                'apt-get update',
                'apt-get install -y ca-certificates curl gnupg',
                'install -m 0755 -d /etc/apt/keyrings',
                'curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc',
                'chmod a+r /etc/apt/keyrings/docker.asc',
                'echo \
                  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
                  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
                  tee /etc/apt/sources.list.d/docker.list > /dev/null',
                'apt-get update',
                'apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin',
            ]
            for cmd in commands:
                try:
                    subprocess.run(
                        cmd, shell=True, check=True, stdout=subprocess.DEVNULL
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f'Image build failed:\n{e}')
                    logger.error(f'Command output:\n{e.output}')
                    raise
            logger.info('Downloaded and installed docker binary')

        target_image_hash_name = tags[0]
        target_image_repo, target_image_source_tag = target_image_hash_name.split(':')
        target_image_tag = tags[1].split(':')[1] if len(tags) > 1 else None

        buildx_cmd = [
            'docker' if not self.is_podman else 'podman',
            'buildx',
            'build',
            '--progress=plain',
            f'--build-arg=OPENHANDS_RUNTIME_VERSION={get_version()}',
            f'--build-arg=OPENHANDS_RUNTIME_BUILD_TIME={datetime.datetime.now().isoformat()}',
            f'--tag={target_image_hash_name}',
            '--load',
        ]

        # Include the platform argument only if platform is specified
        if platform:
            buildx_cmd.append(f'--platform={platform}')

        cache_dir = '/tmp/.buildx-cache'
        if use_local_cache and self._is_cache_usable(cache_dir):
            buildx_cmd.extend(
                [
                    f'--cache-from=type=local,src={cache_dir}',
                    f'--cache-to=type=local,dest={cache_dir},mode=max',
                ]
            )

        if extra_build_args:
            buildx_cmd.extend(extra_build_args)

        buildx_cmd.append(path)  # must be last!

        self.rolling_logger.start(
            f'================ {buildx_cmd[0].upper()} BUILD STARTED ================'
        )

        builder_cmd = ['docker', 'buildx', 'use', 'default']
        subprocess.Popen(
            builder_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        try:
            process = subprocess.Popen(
                buildx_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            output_lines = []
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    line = line.strip()
                    if line:
                        output_lines.append(line)  # Store all output lines
                        self._output_logs(line)

            return_code = process.wait()

            if return_code != 0:
                # Use the collected output for error reporting
                output_str = '\n'.join(output_lines)
                raise subprocess.CalledProcessError(
                    return_code,
                    process.args,
                    output=output_str,  # Use the collected output
                    stderr=None,
                )

        except subprocess.CalledProcessError as e:
            logger.error(f'Image build failed with exit code {e.returncode}')
            if e.output:
                logger.error(f'Command output:\n{e.output}')
            elif self.rolling_logger.is_enabled() and self.rolling_logger.all_lines:
                logger.error(f'Docker build output:\n{self.rolling_logger.all_lines}')
            raise

        except subprocess.TimeoutExpired:
            logger.error('Image build timed out')
            raise

        except FileNotFoundError as e:
            logger.error(f'Python executable not found: {e}')
            raise

        except PermissionError as e:
            logger.error(
                f'Permission denied when trying to execute the build command:\n{e}'
            )
            raise

        except Exception as e:
            logger.error(f'An unexpected error occurred during the build process: {e}')
            raise

        logger.info(f'Image [{target_image_hash_name}] build finished.')

        if target_image_tag:
            image = self.docker_client.images.get(target_image_hash_name)
            image.tag(target_image_repo, target_image_tag)
            logger.info(
                f'Re-tagged image [{target_image_hash_name}] with more generic tag [{target_image_tag}]'
            )

        # Check if the image is built successfully
        image = self.docker_client.images.get(target_image_hash_name)
        if image is None:
            raise AgentRuntimeBuildError(
                f'Build failed: Image {target_image_hash_name} not found'
            )

        tags_str = (
            f'{target_image_source_tag}, {target_image_tag}'
            if target_image_tag
            else target_image_source_tag
        )
        logger.info(
            f'Image {target_image_repo} with tags [{tags_str}] built successfully'
        )
        return target_image_hash_name

    def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
        """Check if the image exists in the registry (try to pull it first) or in the local store.

        Args:
            image_name (str): The Docker image to check (<image repo>:<image tag>)
            pull_from_repo (bool): Whether to pull from the remote repo if the image not present locally
        Returns:
            bool: Whether the Docker image exists in the registry or in the local store
        """
        if not image_name:
            logger.error(f'Invalid image name: `{image_name}`')
            return False

        try:
            logger.debug(f'Checking, if image exists locally:\n{image_name}')
            self.docker_client.images.get(image_name)
            logger.debug('Image found locally.')
            return True
        except docker.errors.ImageNotFound:
            if not pull_from_repo:
                logger.debug(
                    f'Image {image_name} {colorize("not found", TermColor.WARNING)} locally'
                )
                return False
            try:
                logger.debug(
                    'Image not found locally. Trying to pull it, please wait...'
                )

                layers: dict[str, dict[str, str]] = {}
                previous_layer_count = 0

                if ':' in image_name:
                    image_repo, image_tag = image_name.split(':', 1)
                else:
                    image_repo = image_name
                    image_tag = None

                for line in self.docker_client.api.pull(
                    image_repo, tag=image_tag, stream=True, decode=True
                ):
                    self._output_build_progress(line, layers, previous_layer_count)
                    previous_layer_count = len(layers)
                logger.debug('Image pulled')
                return True
            except docker.errors.ImageNotFound:
                logger.debug('Could not find image locally or in registry.')
                return False
            except Exception as e:
                msg = f'Image {colorize("could not be pulled", TermColor.ERROR)}: '
                ex_msg = str(e)
                if 'Not Found' in ex_msg:
                    msg += 'image not found in registry.'
                else:
                    msg += f'{ex_msg}'
                logger.debug(msg)
                return False

    def _output_logs(self, new_line: str) -> None:
        if not self.rolling_logger.is_enabled():
            logger.debug(new_line)
        else:
            self.rolling_logger.add_line(new_line)

    def _output_build_progress(
        self, current_line: dict, layers: dict, previous_layer_count: int
    ) -> None:
        if 'id' in current_line and 'progressDetail' in current_line:
            layer_id = current_line['id']
            if layer_id not in layers:
                layers[layer_id] = {'status': '', 'progress': '', 'last_logged': 0}

            if 'status' in current_line:
                layers[layer_id]['status'] = current_line['status']

            if 'progress' in current_line:
                layers[layer_id]['progress'] = current_line['progress']

            if 'progressDetail' in current_line:
                progress_detail = current_line['progressDetail']
                if 'total' in progress_detail and 'current' in progress_detail:
                    total = progress_detail['total']
                    current = progress_detail['current']
                    percentage = min(
                        (current / total) * 100, 100
                    )  # Ensure it doesn't exceed 100%
                else:
                    percentage = (
                        100 if layers[layer_id]['status'] == 'Download complete' else 0
                    )

            if self.rolling_logger.is_enabled():
                self.rolling_logger.move_back(previous_layer_count)
                for lid, layer_data in sorted(layers.items()):
                    self.rolling_logger.replace_current_line()
                    status = layer_data['status']
                    progress = layer_data['progress']
                    if status == 'Download complete':
                        self.rolling_logger.write_immediately(
                            f'Layer {lid}: Download complete'
                        )
                    elif status == 'Already exists':
                        self.rolling_logger.write_immediately(
                            f'Layer {lid}: Already exists'
                        )
                    else:
                        self.rolling_logger.write_immediately(
                            f'Layer {lid}: {progress} {status}'
                        )
            elif percentage != 0 and (
                percentage - layers[layer_id]['last_logged'] >= 10 or percentage == 100
            ):
                logger.debug(
                    f'Layer {layer_id}: {layers[layer_id]["progress"]} {layers[layer_id]["status"]}'
                )

            layers[layer_id]['last_logged'] = percentage
        elif 'status' in current_line:
            logger.debug(current_line['status'])

    def _prune_old_cache_files(self, cache_dir: str, max_age_days: int = 7) -> None:
        """Prune cache files older than the specified number of days.

        Args:
            cache_dir (str): The path to the cache directory.
            max_age_days (int): The maximum age of cache files in days.
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60

            for root, _, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            logger.debug(f'Removed old cache file: {file_path}')
                    except Exception as e:
                        logger.warning(f'Error processing cache file {file_path}: {e}')
        except Exception as e:
            logger.warning(f'Error during build cache pruning: {e}')

    def _is_cache_usable(self, cache_dir: str) -> bool:
        """Check if the cache directory is usable (exists and is writable).

        Args:
            cache_dir (str): The path to the cache directory.

        Returns:
            bool: True if the cache directory is usable, False otherwise.
        """
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
                logger.debug(f'Created cache directory: {cache_dir}')
            except OSError as e:
                logger.debug(f'Failed to create cache directory {cache_dir}: {e}')
                return False

        if not os.access(cache_dir, os.W_OK):
            logger.warning(
                f'Cache directory {cache_dir} is not writable. Caches will not be used for Docker builds.'
            )
            return False

        self._prune_old_cache_files(cache_dir)

        logger.debug(f'Cache directory {cache_dir} is usable')
        return True
