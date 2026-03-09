# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from functools import lru_cache
from typing import Callable
from uuid import UUID

import tenacity
import yaml
from kubernetes import client, config
from kubernetes.client.models import (
    V1Container,
    V1ContainerPort,
    V1EnvVar,
    V1HTTPIngressPath,
    V1HTTPIngressRuleValue,
    V1Ingress,
    V1IngressBackend,
    V1IngressRule,
    V1IngressServiceBackend,
    V1IngressSpec,
    V1IngressTLS,
    V1ObjectMeta,
    V1PersistentVolumeClaim,
    V1PersistentVolumeClaimSpec,
    V1PersistentVolumeClaimVolumeSource,
    V1Pod,
    V1PodSpec,
    V1ResourceRequirements,
    V1SecurityContext,
    V1Service,
    V1ServiceBackendPort,
    V1ServicePort,
    V1ServiceSpec,
    V1Toleration,
    V1Volume,
    V1VolumeMount,
)

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import DEBUG
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils.command import get_action_execution_server_startup_command
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import add_shutdown_listener
from openhands.utils.tenacity_stop import stop_if_should_exit

POD_NAME_PREFIX = 'openhands-runtime-'
POD_LABEL = 'openhands-runtime'


class KubernetesRuntime(ActionExecutionClient):
    """A Kubernetes runtime for OpenHands that works with Kind.

    This runtime creates pods in a Kubernetes cluster to run the agent code.
    It uses the Kubernetes Python client to create and manage the pods.

    Args:
        config (OpenHandsConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
        status_callback (Callable | None, optional): Callback for status updates. Defaults to None.
        attach_to_existing (bool, optional): Whether to attach to an existing pod. Defaults to False.
        headless_mode (bool, optional): Whether to run in headless mode. Defaults to True.
    """

    _shutdown_listener_id: UUID | None = None
    _namespace: str = ''

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    ):
        if not KubernetesRuntime._shutdown_listener_id:
            KubernetesRuntime._shutdown_listener_id = add_shutdown_listener(
                lambda: KubernetesRuntime._cleanup_k8s_resources(
                    namespace=self._k8s_namespace,
                    remove_pvc=True,
                    conversation_id=self.sid,
                )  # this is when you ctrl+c.
            )
        self.config = config
        self._runtime_initialized: bool = False
        self.status_callback = status_callback

        # Load and validate Kubernetes configuration
        if self.config.kubernetes is None:
            raise ValueError(
                'Kubernetes configuration is required when using KubernetesRuntime. '
                'Please add a [kubernetes] section to your configuration.'
            )

        self._k8s_config = self.config.kubernetes
        self._k8s_namespace = self._k8s_config.namespace
        KubernetesRuntime._namespace = self._k8s_namespace

        # Initialize ports with default values in the required range
        self._container_port = 8080  # Default internal container port
        self._vscode_port = 8081  # Default VSCode port.
        self._app_ports: list[int] = [
            30082,
            30083,
        ]  # Default app ports in valid range # The agent prefers these when exposing an application.

        self.k8s_client, self.k8s_networking_client = self._init_kubernetes_client()

        self.pod_image = self.config.sandbox.runtime_container_image
        if not self.pod_image:
            # If runtime_container_image isn't set, use the base_container_image as a fallback
            self.pod_image = self.config.sandbox.base_container_image

        self.pod_name = POD_NAME_PREFIX + sid

        # Initialize the API URL with the initial port value
        self.k8s_local_url = f'http://{self._get_svc_name(self.pod_name)}.{self._k8s_namespace}.svc.cluster.local'
        self.api_url = f'{self.k8s_local_url}:{self._container_port}'

        super().__init__(
            config,
            event_stream,
            llm_registry,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
            user_id,
            git_provider_tokens,
        )

    @staticmethod
    def _get_svc_name(pod_name: str) -> str:
        """Get the service name for the pod."""
        return f'{pod_name}-svc'

    @staticmethod
    def _get_vscode_svc_name(pod_name: str) -> str:
        """Get the VSCode service name for the pod."""
        return f'{pod_name}-svc-code'

    @staticmethod
    def _get_vscode_ingress_name(pod_name: str) -> str:
        """Get the VSCode ingress name for the pod."""
        return f'{pod_name}-ingress-code'

    @staticmethod
    def _get_vscode_tls_secret_name(pod_name: str) -> str:
        """Get the TLS secret name for the VSCode ingress."""
        return f'{pod_name}-tls-secret'

    @staticmethod
    def _get_pvc_name(pod_name: str) -> str:
        """Get the PVC name for the pod."""
        return f'{pod_name}-pvc'

    @staticmethod
    def _get_pod_name(sid: str) -> str:
        """Get the pod name for the session."""
        return POD_NAME_PREFIX + sid

    @property
    def action_execution_server_url(self):
        return self.api_url

    @property
    def node_selector(self) -> dict[str, str] | None:
        if (
            not self._k8s_config.node_selector_key
            or not self._k8s_config.node_selector_val
        ):
            return None
        return {self._k8s_config.node_selector_key: self._k8s_config.node_selector_val}

    @property
    def tolerations(self) -> list[V1Toleration] | None:
        if not self._k8s_config.tolerations_yaml:
            return None
        tolerations_yaml_str = self._k8s_config.tolerations_yaml
        tolerations = []
        try:
            tolerations_data = yaml.safe_load(tolerations_yaml_str)
            if isinstance(tolerations_data, list):
                for toleration in tolerations_data:
                    tolerations.append(V1Toleration(**toleration))
            else:
                logger.error(
                    f'Invalid tolerations format. Should be type list: {tolerations_yaml_str}. Expected a list.'
                )
                return None
        except yaml.YAMLError as e:
            logger.error(
                f'Error parsing tolerations YAML: {tolerations_yaml_str}. Error: {e}'
            )
            return None
        return tolerations

    async def connect(self):
        """Connect to the runtime by creating or attaching to a pod."""
        self.log('info', f'Connecting to runtime with conversation ID: {self.sid}')
        self.log('info', f'self._attach_to_existing: {self.attach_to_existing}')
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        self.log('info', f'Using API URL {self.api_url}')

        try:
            await call_sync_from_async(self._attach_to_pod)
        except client.rest.ApiException as e:
            # we are not set to attach to existing, ignore error and init k8s resources.
            if self.attach_to_existing:
                self.log(
                    'error',
                    f'Pod {self.pod_name} not found or cannot connect to it.',
                )
                raise AgentRuntimeDisconnectedError from e

            self.log('info', f'Starting runtime with image: {self.pod_image}')
            try:
                await call_sync_from_async(self._init_k8s_resources)
                self.log(
                    'info',
                    f'Pod started: {self.pod_name}. VSCode URL: {self.vscode_url}',
                )
            except Exception as init_error:
                self.log('error', f'Failed to initialize k8s resources: {init_error}')
                raise AgentRuntimeNotFoundError(
                    f'Failed to initialize kubernetes resources: {init_error}'
                ) from init_error

        if not self.attach_to_existing:
            self.log('info', 'Waiting for pod to become ready ...')
            self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        try:
            await call_sync_from_async(self._wait_until_ready)
        except Exception as alive_error:
            self.log('error', f'Failed to connect to runtime: {alive_error}')
            self.set_runtime_status(
                RuntimeStatus.ERROR_RUNTIME_DISCONNECTED,
                f'Failed to connect to runtime: {alive_error}',
            )
            raise AgentRuntimeDisconnectedError(
                f'Failed to connect to runtime: {alive_error}'
            ) from alive_error

        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.log(
            'info',
            f'Pod initialized with plugins: {[plugin.name for plugin in self.plugins]}. VSCode URL: {self.vscode_url}',
        )
        if not self.attach_to_existing:
            self.set_runtime_status(RuntimeStatus.READY)
        self._runtime_initialized = True

    def _attach_to_pod(self):
        """Attach to an existing pod."""
        try:
            pod = self.k8s_client.read_namespaced_pod(
                name=self.pod_name, namespace=self._k8s_namespace
            )

            if pod.status.phase != 'Running':
                try:
                    self._wait_until_ready()
                except TimeoutError:
                    raise AgentRuntimeDisconnectedError(
                        f'Pod {self.pod_name} exists but failed to become ready.'
                    )

            self.log('info', f'Successfully attached to pod {self.pod_name}')
            return True

        except client.rest.ApiException as e:
            self.log('error', f'Failed to attach to pod: {e}')
            raise

    @tenacity.retry(
        stop=tenacity.stop_after_delay(300) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception_type(TimeoutError),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def _wait_until_ready(self):
        """Wait until the runtime server is alive by checking the pod status in Kubernetes."""
        self.log('info', f'Checking if pod {self.pod_name} is ready in Kubernetes')
        pod = self.k8s_client.read_namespaced_pod(
            name=self.pod_name, namespace=self._k8s_namespace
        )
        if pod.status.phase == 'Running' and pod.status.conditions:
            for condition in pod.status.conditions:
                if condition.type == 'Ready' and condition.status == 'True':
                    self.log('info', f'Pod {self.pod_name} is ready!')
                    return True  # Exit the function if the pod is ready

        self.log(
            'info',
            f'Pod {self.pod_name} is not ready yet. Current phase: {pod.status.phase}',
        )
        raise TimeoutError(f'Pod {self.pod_name} is not in Running state yet.')

    @staticmethod
    @lru_cache(maxsize=1)
    def _init_kubernetes_client() -> tuple[client.CoreV1Api, client.NetworkingV1Api]:
        """Initialize the Kubernetes client."""
        try:
            config.load_incluster_config()  # Even local usage with mirrord technically uses an incluster config.
            return client.CoreV1Api(), client.NetworkingV1Api()
        except Exception as ex:
            logger.error(
                'Failed to initialize Kubernetes client. Make sure you have kubectl configured correctly or are running in a Kubernetes cluster.',
            )
            raise ex

    @staticmethod
    def _cleanup_k8s_resources(
        namespace: str, remove_pvc: bool = False, conversation_id: str = ''
    ):
        """Clean up Kubernetes resources with our prefix in the namespace.

        :param remove_pvc: If True, also remove persistent volume claims (defaults to False).
        """
        try:
            k8s_api, k8s_networking_api = KubernetesRuntime._init_kubernetes_client()

            pod_name = KubernetesRuntime._get_pod_name(conversation_id)
            service_name = KubernetesRuntime._get_svc_name(pod_name)
            vscode_service_name = KubernetesRuntime._get_vscode_svc_name(pod_name)
            ingress_name = KubernetesRuntime._get_vscode_ingress_name(pod_name)
            pvc_name = KubernetesRuntime._get_pvc_name(pod_name)

            try:
                if remove_pvc:
                    # Delete PVC if requested
                    k8s_api.delete_namespaced_persistent_volume_claim(
                        name=pvc_name,
                        namespace=namespace,
                        body=client.V1DeleteOptions(),
                    )
                    logger.info(f'Deleted PVC {pvc_name}')

                k8s_api.delete_namespaced_pod(
                    name=pod_name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(),
                )
                logger.info(f'Deleted pod {pod_name}')

                k8s_api.delete_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                )
                logger.info(f'Deleted service {service_name}')
                # Delete the vs code service
                k8s_api.delete_namespaced_service(
                    name=vscode_service_name, namespace=namespace
                )
                logger.info(f'Deleted service {vscode_service_name}')

                k8s_networking_api.delete_namespaced_ingress(
                    name=ingress_name, namespace=namespace
                )
                logger.info(f'Deleted ingress {ingress_name}')
            except client.rest.ApiException:
                # Service might not exist, ignore
                pass
            logger.info('Cleaned up Kubernetes resources')
        except Exception as e:
            logger.error(f'Error cleaning up k8s resources: {e}')

    def _get_pvc_manifest(self):
        """Create a PVC manifest for the runtime pod."""
        # Create PVC
        pvc = V1PersistentVolumeClaim(
            api_version='v1',
            kind='PersistentVolumeClaim',
            metadata=V1ObjectMeta(
                name=self._get_pvc_name(self.pod_name), namespace=self._k8s_namespace
            ),
            spec=V1PersistentVolumeClaimSpec(
                access_modes=['ReadWriteOnce'],
                resources=client.V1ResourceRequirements(
                    requests={'storage': self._k8s_config.pvc_storage_size}
                ),
                storage_class_name=self._k8s_config.pvc_storage_class,
            ),
        )

        return pvc

    def _get_vscode_service_manifest(self):
        """Create a service manifest for the VSCode server."""
        vscode_service_spec = V1ServiceSpec(
            selector={'app': POD_LABEL, 'session': self.sid},
            type='ClusterIP',
            ports=[
                V1ServicePort(
                    port=self._vscode_port,
                    target_port='vscode',
                    name='code',
                )
            ],
        )

        vscode_service = V1Service(
            metadata=V1ObjectMeta(name=self._get_vscode_svc_name(self.pod_name)),
            spec=vscode_service_spec,
        )
        return vscode_service

    def _get_runtime_service_manifest(self):
        """Create a service manifest for the runtime pod execution-server."""
        service_spec = V1ServiceSpec(
            selector={'app': POD_LABEL, 'session': self.sid},
            type='ClusterIP',
            ports=[
                V1ServicePort(
                    port=self._container_port,
                    target_port='http',
                    name='execution-server',
                )
            ],
        )

        service = V1Service(
            metadata=V1ObjectMeta(name=self._get_svc_name(self.pod_name)),
            spec=service_spec,
        )
        return service

    def _get_runtime_pod_manifest(self):
        """Create a pod manifest for the runtime sandbox."""
        # Prepare environment variables
        environment = [
            V1EnvVar(name='port', value=str(self._container_port)),
            V1EnvVar(name='PYTHONUNBUFFERED', value='1'),
            V1EnvVar(name='VSCODE_PORT', value=str(self._vscode_port)),
        ]

        if self.config.debug or DEBUG:
            environment.append(V1EnvVar(name='DEBUG', value='true'))

        # Add runtime startup env vars
        for key, value in self.config.sandbox.runtime_startup_env_vars.items():
            environment.append(V1EnvVar(name=key, value=value))

        # Prepare volume mounts if workspace is configured
        volume_mounts = [
            V1VolumeMount(
                name='workspace-volume',
                mount_path=self.config.workspace_mount_path_in_sandbox,
            ),
        ]
        volumes = [
            V1Volume(
                name='workspace-volume',
                persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                    claim_name=self._get_pvc_name(self.pod_name)
                ),
            )
        ]

        # Prepare container ports
        container_ports = [
            V1ContainerPort(container_port=self._container_port, name='http'),
        ]

        if self.vscode_enabled:
            container_ports.append(
                V1ContainerPort(container_port=self._vscode_port, name='vscode')
            )

        for port in self._app_ports:
            container_ports.append(V1ContainerPort(container_port=port))

        # Define the readiness probe
        health_check = client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path='/alive',
                port=self._container_port,  # Or the port your application listens on
            ),
            initial_delay_seconds=5,  # Adjust as needed
            period_seconds=10,  # Adjust as needed
            timeout_seconds=5,  # Adjust as needed
            success_threshold=1,
            failure_threshold=3,
        )
        # Prepare command
        # Entry point command for generated sandbox runtime pod.
        command = get_action_execution_server_startup_command(
            server_port=self._container_port,
            plugins=self.plugins,
            app_config=self.config,
            override_user_id=0,  # if we use the default of app_config.run_as_openhands then we cant edit files in vscode due to file perms.
            override_username='root',
        )

        # Prepare resource requirements based on config
        resources = V1ResourceRequirements(
            limits={'memory': self._k8s_config.resource_memory_limit},
            requests={
                'cpu': self._k8s_config.resource_cpu_request,
                'memory': self._k8s_config.resource_memory_request,
            },
        )

        # Set security context for the container
        security_context = V1SecurityContext(privileged=self._k8s_config.privileged)

        # Create the container definition
        container = V1Container(
            name='runtime',
            image=self.pod_image,
            command=command,
            env=environment,
            ports=container_ports,
            volume_mounts=volume_mounts,
            working_dir='/openhands/code/',
            resources=resources,
            readiness_probe=health_check,
            security_context=security_context,
        )

        # Create the pod definition
        image_pull_secrets = None
        if self._k8s_config.image_pull_secret:
            image_pull_secrets = [
                client.V1LocalObjectReference(name=self._k8s_config.image_pull_secret)
            ]
        pod = V1Pod(
            metadata=V1ObjectMeta(
                name=self.pod_name, labels={'app': POD_LABEL, 'session': self.sid}
            ),
            spec=V1PodSpec(
                containers=[container],
                volumes=volumes,
                restart_policy='Never',
                image_pull_secrets=image_pull_secrets,
                node_selector=self.node_selector,
                tolerations=self.tolerations,
            ),
        )

        return pod

    def _get_vscode_ingress_manifest(self):
        """Create an ingress manifest for the VSCode server."""
        tls = []
        if self._k8s_config.ingress_tls_secret:
            runtime_tls = V1IngressTLS(
                hosts=[self.ingress_domain],
                secret_name=self._k8s_config.ingress_tls_secret,
            )
            tls = [runtime_tls]

        rules = [
            V1IngressRule(
                host=self.ingress_domain,
                http=V1HTTPIngressRuleValue(
                    paths=[
                        V1HTTPIngressPath(
                            path='/',
                            path_type='Prefix',
                            backend=V1IngressBackend(
                                service=V1IngressServiceBackend(
                                    port=V1ServiceBackendPort(
                                        number=self._vscode_port,
                                    ),
                                    name=self._get_vscode_svc_name(self.pod_name),
                                )
                            ),
                        )
                    ]
                ),
            )
        ]
        ingress_spec = V1IngressSpec(rules=rules, tls=tls)

        ingress = V1Ingress(
            api_version='networking.k8s.io/v1',
            metadata=V1ObjectMeta(
                name=self._get_vscode_ingress_name(self.pod_name),
                annotations={
                    'external-dns.alpha.kubernetes.io/hostname': self.ingress_domain
                },
            ),
            spec=ingress_spec,
        )

        return ingress

    def _pvc_exists(self):
        """Check if the PVC already exists."""
        try:
            pvc = self.k8s_client.read_namespaced_persistent_volume_claim(
                name=self._get_pvc_name(self.pod_name), namespace=self._k8s_namespace
            )
            return pvc is not None
        except client.rest.ApiException as e:
            if e.status == 404:
                return False
            self.log('error', f'Error checking PVC existence: {e}')

    def _init_k8s_resources(self):
        """Initialize the Kubernetes resources."""
        self.log('info', 'Preparing to start pod...')
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        self.log('info', f'Runtime will be accessible at {self.api_url}')

        pod = self._get_runtime_pod_manifest()
        service = self._get_runtime_service_manifest()
        vscode_service = self._get_vscode_service_manifest()
        pvc_manifest = self._get_pvc_manifest()
        ingress = self._get_vscode_ingress_manifest()

        # Create the pod in Kubernetes
        try:
            if not self._pvc_exists():
                # Create PVC if it doesn't exist
                self.k8s_client.create_namespaced_persistent_volume_claim(
                    namespace=self._k8s_namespace, body=pvc_manifest
                )
                self.log('info', f'Created PVC {self._get_pvc_name(self.pod_name)}')
            self.k8s_client.create_namespaced_pod(
                namespace=self._k8s_namespace, body=pod
            )
            self.log('info', f'Created pod {self.pod_name}.')
            # Create a service to expose the pod for external access
            self.k8s_client.create_namespaced_service(
                namespace=self._k8s_namespace, body=service
            )
            self.log('info', f'Created service {self._get_svc_name(self.pod_name)}')

            # Create second service service for the vscode server.
            self.k8s_client.create_namespaced_service(
                namespace=self._k8s_namespace, body=vscode_service
            )
            self.log(
                'info', f'Created service {self._get_vscode_svc_name(self.pod_name)}'
            )

            # create the vscode ingress.
            self.k8s_networking_client.create_namespaced_ingress(
                namespace=self._k8s_namespace, body=ingress
            )
            self.log(
                'info',
                f'Created ingress {self._get_vscode_ingress_name(self.pod_name)}',
            )

            # Wait for the pod to be running
            self._wait_until_ready()

        except client.rest.ApiException as e:
            self.log('error', f'Failed to create pod and services: {e}')
            raise
        except RuntimeError as e:
            self.log('error', f'Port forwarding failed: {e}')
            raise

    def close(self):
        """Close the runtime and clean up resources."""
        # this is called when a single conversation question is answered or a tab is closed.
        self.log(
            'info',
            f'Closing runtime and cleaning up resources for conersation ID: {self.sid}',
        )
        # Call parent class close method first
        super().close()

        # Return early if we should keep the runtime alive or if we're attaching to existing
        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            self.log(
                'info', 'Keeping runtime alive due to configuration or attach mode'
            )
            return

        try:
            self._cleanup_k8s_resources(
                namespace=self._k8s_namespace,
                remove_pvc=False,
                conversation_id=self.sid,
            )
        except Exception as e:
            self.log('error', f'Error closing runtime: {e}')

    @property
    def ingress_domain(self) -> str:
        """Get the ingress domain for the runtime."""
        return f'{self.sid}.{self._k8s_config.ingress_domain}'

    @property
    def vscode_url(self) -> str | None:
        """Get the URL for VSCode server if enabled."""
        if not self.vscode_enabled:
            return None
        token = super().get_vscode_token()
        if not token:
            return None

        protocol = 'https' if self._k8s_config.ingress_tls_secret else 'http'
        vscode_url = f'{protocol}://{self.ingress_domain}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        self.log('info', f'VSCode URL: {vscode_url}')
        return vscode_url

    @property
    def web_hosts(self) -> dict[str, int]:
        """Get web hosts dict mapping for browser access."""
        hosts = {}
        for idx, port in enumerate(self._app_ports):
            hosts[f'{self.k8s_local_url}:{port}'] = port
        return hosts

    @classmethod
    async def delete(cls, conversation_id: str):
        """Delete resources associated with a conversation."""
        # This is triggered when you actually do the delete in the UI on the conversation.
        try:
            cls._cleanup_k8s_resources(
                namespace=cls._namespace,
                remove_pvc=True,
                conversation_id=conversation_id,
            )

        except Exception as e:
            logger.error(
                f'Error deleting resources for conversation {conversation_id}: {e}'
            )
