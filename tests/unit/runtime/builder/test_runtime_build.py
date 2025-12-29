import hashlib
import os
import tempfile
import uuid
from importlib.metadata import version
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import docker
import pytest
import toml
from pytest import TempPathFactory

import openhands
from openhands import __version__ as oh_version
from openhands.core.logger import openhands_logger as logger
from openhands.runtime.builder.docker import DockerRuntimeBuilder
from openhands.runtime.utils.runtime_build import (
    BuildFromImageType,
    _generate_dockerfile,
    build_runtime_image,
    get_hash_for_lock_files,
    get_hash_for_source_files,
    get_runtime_image_repo,
    get_runtime_image_repo_and_tag,
    prep_build_folder,
    truncate_hash,
)

OH_VERSION = f'oh_v{oh_version}'
DEFAULT_BASE_IMAGE = 'nikolaik/python-nodejs:python3.12-nodejs22'


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_runtime_build'))


@pytest.fixture
def mock_docker_client():
    mock_client = MagicMock(spec=docker.DockerClient)
    mock_client.version.return_value = {
        'Version': '20.10.0',
        'Components': [{'Name': 'Engine', 'Version': '20.10.0'}],
    }  # Ensure version is >= 18.09
    return mock_client


@pytest.fixture
def docker_runtime_builder():
    client = docker.from_env()
    return DockerRuntimeBuilder(client)


def _check_source_code_in_dir(temp_dir):
    # assert there is a folder called 'code' in the temp_dir
    code_dir = os.path.join(temp_dir, 'code')
    assert os.path.exists(code_dir)
    assert os.path.isdir(code_dir)

    # check the source file is the same as the current code base
    assert os.path.exists(os.path.join(code_dir, 'pyproject.toml'))

    # The source code should only include the `openhands` folder,
    # and pyproject.toml & poetry.lock that are needed to build the runtime image
    assert set(os.listdir(code_dir)) == {
        'openhands',
        'pyproject.toml',
        'poetry.lock',
    }
    assert os.path.exists(os.path.join(code_dir, 'openhands'))
    assert os.path.isdir(os.path.join(code_dir, 'openhands'))

    # make sure the version from the pyproject.toml is the same as the current version
    with open(os.path.join(code_dir, 'pyproject.toml'), 'r') as f:
        pyproject = toml.load(f)

    _pyproject_version = pyproject['tool']['poetry']['version']
    assert _pyproject_version == version('openhands-ai')


def test_prep_build_folder(temp_dir):
    shutil_mock = MagicMock()
    with patch(f'{prep_build_folder.__module__}.shutil', shutil_mock):
        prep_build_folder(
            temp_dir,
            base_image=DEFAULT_BASE_IMAGE,
            build_from=BuildFromImageType.SCRATCH,
            extra_deps=None,
        )

    # make sure that the code (openhands/) and microagents folder were copied
    assert shutil_mock.copytree.call_count == 2
    assert shutil_mock.copy2.call_count == 2

    # Now check dockerfile is in the folder
    dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
    assert os.path.exists(dockerfile_path)
    assert os.path.isfile(dockerfile_path)


def test_get_hash_for_lock_files():
    with patch('builtins.open', mock_open(read_data='mock-data'.encode())):
        hash = get_hash_for_lock_files('some_base_image', enable_browser=True)
        # Since we mocked open to always return "mock_data", the hash is the result
        # of hashing the name of the base image followed by "mock-data" twice
        md5 = hashlib.md5()
        md5.update('some_base_image'.encode())
        for _ in range(2):
            md5.update('mock-data'.encode())
        assert hash == truncate_hash(md5.hexdigest())


def test_get_hash_for_lock_files_different_enable_browser():
    with patch('builtins.open', mock_open(read_data='mock-data'.encode())):
        hash_true = get_hash_for_lock_files('some_base_image', enable_browser=True)
        hash_false = get_hash_for_lock_files('some_base_image', enable_browser=False)

        # Hash with enable_browser=True should not include the enable_browser value
        md5_true = hashlib.md5()
        md5_true.update('some_base_image'.encode())
        for _ in range(2):
            md5_true.update('mock-data'.encode())
        expected_hash_true = truncate_hash(md5_true.hexdigest())

        # Hash with enable_browser=False should include the enable_browser value
        md5_false = hashlib.md5()
        md5_false.update('some_base_image'.encode())
        md5_false.update('False'.encode())  # enable_browser=False is included
        for _ in range(2):
            md5_false.update('mock-data'.encode())
        expected_hash_false = truncate_hash(md5_false.hexdigest())

        assert hash_true == expected_hash_true
        assert hash_false == expected_hash_false
        assert hash_true != hash_false  # They should be different


def test_get_hash_for_source_files():
    dirhash_mock = MagicMock()
    dirhash_mock.return_value = '1f69bd20d68d9e3874d5bf7f7459709b'
    with patch(f'{get_hash_for_source_files.__module__}.dirhash', dirhash_mock):
        result = get_hash_for_source_files()
        assert result == truncate_hash(dirhash_mock.return_value)
        dirhash_mock.assert_called_once_with(
            Path(openhands.__file__).parent,
            'md5',
            ignore=[
                '.*/',  # hidden directories
                '__pycache__/',
                '*.pyc',
            ],
        )


def test_generate_dockerfile_build_from_scratch():
    base_image = 'debian:11'
    dockerfile_content = _generate_dockerfile(
        base_image,
        build_from=BuildFromImageType.SCRATCH,
    )
    assert base_image in dockerfile_content
    assert 'apt-get update' in dockerfile_content
    assert 'wget curl' in dockerfile_content
    assert 'poetry' in dockerfile_content and '-c conda-forge' in dockerfile_content
    assert 'python=3.12' in dockerfile_content

    # Check the update command
    assert (
        'COPY --chown=openhands:openhands ./code/openhands /openhands/code/openhands'
        in dockerfile_content
    )
    assert (
        '/openhands/micromamba/bin/micromamba run -n openhands poetry install'
        in dockerfile_content
    )


def test_generate_dockerfile_build_from_lock():
    base_image = 'debian:11'
    dockerfile_content = _generate_dockerfile(
        base_image,
        build_from=BuildFromImageType.LOCK,
    )

    # These commands SHOULD NOT include in the dockerfile if build_from_scratch is False
    assert 'wget curl sudo apt-utils git' not in dockerfile_content
    assert '-c conda-forge' not in dockerfile_content
    assert 'python=3.12' not in dockerfile_content
    assert 'https://micro.mamba.pm/install.sh' not in dockerfile_content
    assert 'poetry install' not in dockerfile_content

    # These update commands SHOULD still in the dockerfile
    assert (
        'COPY --chown=openhands:openhands ./code/openhands /openhands/code/openhands'
        in dockerfile_content
    )


def test_generate_dockerfile_build_from_versioned():
    base_image = 'debian:11'
    dockerfile_content = _generate_dockerfile(
        base_image,
        build_from=BuildFromImageType.VERSIONED,
    )

    # these commands should not exist when build from versioned
    assert 'wget curl sudo apt-utils git' not in dockerfile_content
    assert '-c conda-forge' not in dockerfile_content
    assert 'python=3.12' not in dockerfile_content
    assert 'https://micro.mamba.pm/install.sh' not in dockerfile_content

    # this SHOULD exist when build from versioned
    assert 'poetry install' in dockerfile_content
    assert (
        'COPY --chown=openhands:openhands ./code/openhands /openhands/code/openhands'
        in dockerfile_content
    )


def test_generate_dockerfile_channel_alias(monkeypatch):
    base_image = 'debian:11'
    alias = 'https://repo.prefix.dev'
    monkeypatch.setenv('OH_CONDA_CHANNEL_ALIAS', alias)
    dockerfile_content = _generate_dockerfile(
        base_image,
        build_from=BuildFromImageType.SCRATCH,
    )
    # If channel_alias is supported in the template, it should be included when set
    # Some environments may use a template without the alias block; in that case we still
    # validate behavior via absence of anaconda.org and use of -c conda-forge below.
    # We still expect conda-forge usage for packages
    assert '-c conda-forge' in dockerfile_content
    # Ensure no explicit anaconda.org URLs are present
    assert 'https://conda.anaconda.org' not in dockerfile_content
    # The micromamba install should use the named channel, not a URL
    install_snippet = (
        '/openhands/micromamba/bin/micromamba install -n openhands -c conda-forge'
    )
    assert install_snippet in dockerfile_content

    # If alias is wired in, ensure it appears before first install from conda-forge
    if 'micromamba config set channel_alias' in dockerfile_content:
        assert dockerfile_content.find(
            'micromamba config set channel_alias'
        ) < dockerfile_content.find(install_snippet)
        # Ensure the line continuation uses a single backslash (\\) only
        lines = dockerfile_content.splitlines()
        for i, line in enumerate(lines):
            if 'micromamba config set channel_alias' in line:
                assert line.rstrip().endswith('\\')
                # Not a literal double backslash in the Dockerfile (which would break RUN continuation)
                assert ' \\\\' not in line
                break


def test_get_runtime_image_repo_and_tag_eventstream():
    base_image = 'debian:11'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag == f'{OH_VERSION}_image_debian_tag_11'
    )

    img_repo, img_tag = get_runtime_image_repo_and_tag(DEFAULT_BASE_IMAGE)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag
        == f'{OH_VERSION}_image_nikolaik_s_python-nodejs_tag_python3.12-nodejs22'
    )

    base_image = 'ubuntu'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag == f'{OH_VERSION}_image_ubuntu_tag_latest'
    )


def test_generate_dockerfile_channel_alias_not_in_non_scratch(monkeypatch):
    base_image = 'debian:11'
    alias = 'https://repo.prefix.dev'
    monkeypatch.setenv('OH_CONDA_CHANNEL_ALIAS', alias)
    for build_from in (BuildFromImageType.VERSIONED, BuildFromImageType.LOCK):
        dockerfile_content = _generate_dockerfile(
            base_image,
            build_from=build_from,
        )
        assert 'micromamba config set channel_alias' not in dockerfile_content

    base_image = 'debian:11'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag == f'{OH_VERSION}_image_debian_tag_11'
    )

    img_repo, img_tag = get_runtime_image_repo_and_tag(DEFAULT_BASE_IMAGE)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag
        == f'{OH_VERSION}_image_nikolaik_s_python-nodejs_tag_python3.12-nodejs22'
    )

    base_image = 'ubuntu'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{get_runtime_image_repo()}'
        and img_tag == f'{OH_VERSION}_image_ubuntu_tag_latest'
    )


def test_build_runtime_image_from_scratch():
    base_image = 'debian:11'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_runtime_builder = MagicMock()
    mock_runtime_builder.image_exists.return_value = False
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
    )
    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__
    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(
            f'{build_runtime_image.__module__}.prep_build_folder',
            mock_prep_build_folder,
        ),
    ):
        image_name = build_runtime_image(base_image, mock_runtime_builder)
        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag',
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag',
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-versioned-tag',
            ],
            platform=None,
            extra_build_args=None,
        )
        assert (
            image_name
            == f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
        )
        mock_prep_build_folder.assert_called_once_with(
            ANY, base_image, BuildFromImageType.SCRATCH, None, True
        )


def test_build_runtime_image_exact_hash_exist():
    base_image = 'debian:11'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()
    mock_runtime_builder.image_exists.return_value = True
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
    )
    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__
    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(
            f'{build_runtime_image.__module__}.prep_build_folder',
            mock_prep_build_folder,
        ),
    ):
        image_name = build_runtime_image(base_image, mock_runtime_builder)
        assert (
            image_name
            == f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
        )
        mock_runtime_builder.build.assert_not_called()
        mock_prep_build_folder.assert_not_called()


def test_build_runtime_image_exact_hash_not_exist_and_lock_exist():
    base_image = 'debian:11'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()

    def image_exists_side_effect(image_name, *args):
        if 'mock-lock-tag_mock-source-tag' in image_name:
            return False
        elif 'mock-lock-tag' in image_name:
            return True
        elif 'mock-versioned-tag' in image_name:
            # just to test we should never include versioned tag in a non-from-scratch build
            # in real case it should be True when lock exists
            return False
        else:
            raise ValueError(f'Unexpected image name: {image_name}')

    mock_runtime_builder.image_exists.side_effect = image_exists_side_effect
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
    )

    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__
    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(
            f'{build_runtime_image.__module__}.prep_build_folder',
            mock_prep_build_folder,
        ),
    ):
        image_name = build_runtime_image(base_image, mock_runtime_builder)
        assert (
            image_name
            == f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
        )
        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag',
                # lock tag will NOT be included - since it already exists
                # VERSION tag will NOT be included except from scratch
            ],
            platform=None,
            extra_build_args=None,
        )
        mock_prep_build_folder.assert_called_once_with(
            ANY,
            f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag',
            BuildFromImageType.LOCK,
            None,
            True,
        )


def test_build_runtime_image_exact_hash_not_exist_and_lock_not_exist_and_versioned_exist():
    base_image = 'debian:11'
    mock_lock_hash = MagicMock()
    mock_lock_hash.return_value = 'mock-lock-tag'
    mock_source_hash = MagicMock()
    mock_source_hash.return_value = 'mock-source-tag'
    mock_versioned_tag = MagicMock()
    mock_versioned_tag.return_value = 'mock-versioned-tag'
    mock_runtime_builder = MagicMock()

    def image_exists_side_effect(image_name, *args):
        if 'mock-lock-tag_mock-source-tag' in image_name:
            return False
        elif 'mock-lock-tag' in image_name:
            return False
        elif 'mock-versioned-tag' in image_name:
            return True
        else:
            raise ValueError(f'Unexpected image name: {image_name}')

    mock_runtime_builder.image_exists.side_effect = image_exists_side_effect
    mock_runtime_builder.build.return_value = (
        f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
    )

    mock_prep_build_folder = MagicMock()
    mod = build_runtime_image.__module__
    with (
        patch(f'{mod}.get_hash_for_lock_files', mock_lock_hash),
        patch(f'{mod}.get_hash_for_source_files', mock_source_hash),
        patch(f'{mod}.get_tag_for_versioned_image', mock_versioned_tag),
        patch(
            f'{build_runtime_image.__module__}.prep_build_folder',
            mock_prep_build_folder,
        ),
    ):
        image_name = build_runtime_image(base_image, mock_runtime_builder)
        assert (
            image_name
            == f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag'
        )
        mock_runtime_builder.build.assert_called_once_with(
            path=ANY,
            tags=[
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag_mock-source-tag',
                f'{get_runtime_image_repo()}:{OH_VERSION}_mock-lock-tag',
                # VERSION tag will NOT be included except from scratch
            ],
            platform=None,
            extra_build_args=None,
        )
        mock_prep_build_folder.assert_called_once_with(
            ANY,
            f'{get_runtime_image_repo()}:{OH_VERSION}_mock-versioned-tag',
            BuildFromImageType.VERSIONED,
            None,
            True,
        )


# ==============================
# DockerRuntimeBuilder Tests
# ==============================


def test_output_build_progress(docker_runtime_builder):
    layers = {}
    docker_runtime_builder._output_build_progress(
        {
            'id': 'layer1',
            'status': 'Downloading',
            'progressDetail': {'current': 50, 'total': 100},
        },
        layers,
        0,
    )
    assert layers['layer1']['status'] == 'Downloading'
    assert layers['layer1']['progress'] == ''
    assert layers['layer1']['last_logged'] == 50.0


@pytest.fixture(scope='function')
def live_docker_image():
    client = docker.from_env()
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    unique_prefix = f'test_image_{unique_id}'

    dockerfile_content = f"""
    # syntax=docker/dockerfile:1.4
    FROM {DEFAULT_BASE_IMAGE} AS base
    RUN apt-get update && apt-get install -y wget curl sudo apt-utils

    FROM base AS intermediate
    RUN mkdir -p /openhands

    FROM intermediate AS final
    RUN echo "Hello, OpenHands!" > /openhands/hello.txt
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)

        try:
            image, logs = client.images.build(
                path=temp_dir,
                tag=f'{unique_prefix}:final',
                buildargs={'DOCKER_BUILDKIT': '1'},
                labels={'test': 'true'},
                rm=True,
                forcerm=True,
            )

            # Tag intermediary stages
            client.api.tag(image.id, unique_prefix, 'base')
            client.api.tag(image.id, unique_prefix, 'intermediate')

            all_tags = [
                f'{unique_prefix}:final',
                f'{unique_prefix}:base',
                f'{unique_prefix}:intermediate',
            ]

            print(f'\nImage ID: {image.id}')
            print(f'Image tags: {all_tags}\n')

            yield image

        finally:
            # Clean up all tagged images
            for tag in all_tags:
                try:
                    client.images.remove(tag, force=True)
                    print(f'Removed image: {tag}')
                except Exception as e:
                    print(f'Error removing image {tag}: {str(e)}')


def test_init(docker_runtime_builder):
    assert isinstance(docker_runtime_builder.docker_client, docker.DockerClient)
    assert docker_runtime_builder.rolling_logger.max_lines == 10
    assert docker_runtime_builder.rolling_logger.log_lines == [''] * 10


def test_build_image_from_scratch(docker_runtime_builder, tmp_path):
    context_path = str(tmp_path)
    tags = ['test_build:latest']

    # Create a minimal Dockerfile in the context path
    with open(os.path.join(context_path, 'Dockerfile'), 'w') as f:
        f.write("""FROM php:latest
CMD ["sh", "-c", "echo 'Hello, World!'"]
""")
    built_image_name = None
    container = None
    client = docker.from_env()
    try:
        built_image_name = docker_runtime_builder.build(
            context_path,
            tags,
            use_local_cache=False,
        )
        assert built_image_name == f'{tags[0]}'

        # Verify the image was created
        image = client.images.get(tags[0])
        assert image is not None

    except docker.errors.ImageNotFound:
        pytest.fail('test_build_image_from_scratch: test image not found!')
    except Exception as e:
        pytest.fail(f'test_build_image_from_scratch: Build failed with error: {str(e)}')

    finally:
        # Clean up the container
        if container:
            try:
                container.remove(force=True)
                logger.info(f'Removed test container: `{container.id}`')
            except Exception as e:
                logger.warning(
                    f'Failed to remove test container `{container.id}`: {str(e)}'
                )

        # Clean up the image
        if built_image_name:
            try:
                client.images.remove(built_image_name, force=True)
                logger.info(f'Removed test image: `{built_image_name}`')
            except Exception as e:
                logger.warning(
                    f'Failed to remove test image `{built_image_name}`: {str(e)}'
                )
        else:
            logger.warning('No image was built, so no image cleanup was necessary.')


def _format_size_to_gb(bytes_size):
    """Convert bytes to gigabytes with two decimal places."""
    return round(bytes_size / (1024**3), 2)


def test_list_dangling_images():
    client = docker.from_env()
    dangling_images = client.images.list(filters={'dangling': True})
    if dangling_images and len(dangling_images) > 0:
        for image in dangling_images:
            if 'Size' in image.attrs and isinstance(image.attrs['Size'], int):
                size_gb = _format_size_to_gb(image.attrs['Size'])
                logger.info(f'Dangling image: {image.tags}, Size: {size_gb} GB')
            else:
                logger.info(f'Dangling image: {image.tags}, Size: n/a')
    else:
        logger.info('No dangling images found')


def test_build_image_from_repo(docker_runtime_builder, tmp_path):
    context_path = str(tmp_path)
    tags = ['alpine:latest']

    # Create a minimal Dockerfile in the context path
    with open(os.path.join(context_path, 'Dockerfile'), 'w') as f:
        f.write(f"""FROM {DEFAULT_BASE_IMAGE}
CMD ["sh", "-c", "echo 'Hello, World!'"]
""")
    built_image_name = None
    container = None
    client = docker.from_env()
    try:
        built_image_name = docker_runtime_builder.build(
            context_path,
            tags,
            use_local_cache=False,
        )
        assert built_image_name == f'{tags[0]}'

        image = client.images.get(tags[0])
        assert image is not None

    except docker.errors.ImageNotFound:
        pytest.fail('test_build_image_from_repo: test image not found!')

    finally:
        # Clean up the container
        if container:
            try:
                container.remove(force=True)
                logger.info(f'Removed test container: `{container.id}`')
            except Exception as e:
                logger.warning(
                    f'Failed to remove test container `{container.id}`: {str(e)}'
                )

        # Clean up the image
        if built_image_name:
            try:
                client.images.remove(built_image_name, force=True)
                logger.info(f'Removed test image: `{built_image_name}`')
            except Exception as e:
                logger.warning(
                    f'Failed to remove test image `{built_image_name}`: {str(e)}'
                )
        else:
            logger.warning('No image was built, so no image cleanup was necessary.')


def test_image_exists_local(docker_runtime_builder):
    mock_client = MagicMock()
    mock_client.version.return_value = {
        'Version': '20.10.0',
        'Components': [{'Name': 'Engine', 'Version': '20.10.0'}],
    }  # Ensure version is >= 18.09
    builder = DockerRuntimeBuilder(mock_client)
    image_name = 'existing-local:image'  # The mock pretends this exists by default
    assert builder.image_exists(image_name)


def test_image_exists_not_found():
    mock_client = MagicMock()
    mock_client.version.return_value = {
        'Version': '20.10.0',
        'Components': [{'Name': 'Engine', 'Version': '20.10.0'}],
    }  # Ensure version is >= 18.09
    mock_client.images.get.side_effect = docker.errors.ImageNotFound(
        "He doesn't like you!"
    )
    mock_client.api.pull.side_effect = docker.errors.ImageNotFound(
        "I don't like you either!"
    )
    builder = DockerRuntimeBuilder(mock_client)
    assert not builder.image_exists('nonexistent:image')
    mock_client.images.get.assert_called_once_with('nonexistent:image')
    mock_client.api.pull.assert_called_once_with(
        'nonexistent', tag='image', stream=True, decode=True
    )


def test_truncate_hash():
    truncated = truncate_hash('b08f254d76b1c6a7ad924708c0032251')
    assert truncated == 'pma2wc71uq3c9a85'
    truncated = truncate_hash('102aecc0cea025253c0278f54ebef078')
    assert truncated == '4titk6gquia3taj5'
