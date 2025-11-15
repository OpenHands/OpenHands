import docker
from openhands.core.logger import openhands_logger as logger


def stop_all_containers(prefix: str) -> None:
    """Stop all containers with the given prefix.

    Uses a fresh Docker client to avoid connection state issues
    in multiprocessing environments.
    """
    docker_client = None
    try:
        docker_client = docker.from_env()
        containers = docker_client.containers.list(all=True)
        for container in containers:
            try:
                if container.name and container.name.startswith(prefix):
                    logger.debug(f'Stopping container: {container.name}')
                    container.stop()
            except docker.errors.APIError as e:
                logger.debug(f'API error stopping container {container.name}: {e}')
                pass
            except docker.errors.NotFound:
                logger.debug(f'Container {container.name} not found (already removed)')
                pass
            except Exception as e:
                logger.warning(f'Unexpected error stopping container {container.name}: {e}')
                pass
    except docker.errors.NotFound:  # yes, this can happen!
        logger.debug('Docker containers not found during cleanup')
        pass
    except Exception as e:
        logger.warning(f'Error during container cleanup: {e}')
        pass
    finally:
        if docker_client:
            try:
                docker_client.close()
            except Exception as e:
                logger.debug(f'Error closing Docker client: {e}')
                pass
