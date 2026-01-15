# DEPRECATED: This module is part of the deprecated 'openhands.runtime' package.
# It will be removed on April 1, 2025. Please migrate to the OpenHands SDK:
# https://github.com/All-Hands-AI/openhands-sdk
import abc


class RuntimeBuilder(abc.ABC):
    @abc.abstractmethod
    def build(
        self,
        path: str,
        tags: list[str],
        platform: str | None = None,
        extra_build_args: list[str] | None = None,
    ) -> str:
        """Build the runtime image.

        Args:
            path (str): The path to the runtime image's build directory.
            tags (list[str]): The tags to apply to the runtime image (e.g., ["repo:my-repo", "sha:my-sha"]).
            platform (str, optional): The target platform for the build. Defaults to None.
            extra_build_args (list[str], optional): Additional build arguments to pass to the builder. Defaults to None.

        Returns:
            str: The name:tag of the runtime image after build (e.g., "repo:sha").
                This can be different from the tags input if the builder chooses to mutate the tags (e.g., adding a
                registry prefix). This should be used for subsequent use (e.g., `docker run`).

        Raises:
            AgentRuntimeBuildError: If the build failed.
        """
        pass

    @abc.abstractmethod
    def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
        """Check if the runtime image exists.

        Args:
            image_name (str): The name of the runtime image (e.g., "repo:sha").
            pull_from_repo (bool): Whether to pull from the remote repo if the image not present locally

        Returns:
            bool: Whether the runtime image exists.
        """
        pass
