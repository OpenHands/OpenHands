import pytest
from pydantic import ValidationError

from openhands.core.config.kubernetes_config import KubernetesConfig


def test_kubernetes_config_defaults():
    """Test that KubernetesConfig has correct default values."""
    config = KubernetesConfig()
    assert config.namespace == 'default'
    assert config.ingress_domain == 'localhost'
    assert config.pvc_storage_size == '2Gi'
    assert config.pvc_storage_class is None
    assert config.resource_cpu_request == '1'
    assert config.resource_memory_request == '1Gi'
    assert config.resource_memory_limit == '2Gi'
    assert config.image_pull_secret is None
    assert config.ingress_tls_secret is None
    assert config.node_selector_key is None
    assert config.node_selector_val is None
    assert config.tolerations_yaml is None
    assert config.kubeconfig_path is None
    assert config.privileged is False


def test_kubernetes_config_custom_values():
    """Test that KubernetesConfig accepts custom values."""
    config = KubernetesConfig(
        namespace='test-ns',
        ingress_domain='test.example.com',
        pvc_storage_size='5Gi',
        pvc_storage_class='fast',
        resource_cpu_request='2',
        resource_memory_request='2Gi',
        resource_memory_limit='4Gi',
        image_pull_secret='pull-secret',
        ingress_tls_secret='tls-secret',
        node_selector_key='zone',
        node_selector_val='us-east-1',
        tolerations_yaml='- key: special\n  value: true',
        kubeconfig_path='/path/to/kubeconfig',
        privileged=True,
    )

    assert config.namespace == 'test-ns'
    assert config.ingress_domain == 'test.example.com'
    assert config.pvc_storage_size == '5Gi'
    assert config.pvc_storage_class == 'fast'
    assert config.resource_cpu_request == '2'
    assert config.resource_memory_request == '2Gi'
    assert config.resource_memory_limit == '4Gi'
    assert config.image_pull_secret == 'pull-secret'
    assert config.ingress_tls_secret == 'tls-secret'
    assert config.node_selector_key == 'zone'
    assert config.node_selector_val == 'us-east-1'
    assert config.tolerations_yaml == '- key: special\n  value: true'
    assert config.kubeconfig_path == '/path/to/kubeconfig'
    assert config.privileged is True


def test_kubernetes_config_validation():
    """Test that KubernetesConfig validates input correctly."""
    # Test that extra fields are not allowed
    with pytest.raises(ValidationError):
        KubernetesConfig(extra_field='not allowed')


def test_kubernetes_config_kubeconfig_path():
    """Test that KubernetesConfig properly handles kubeconfig_path."""
    # Test with kubeconfig_path specified
    config_with_path = KubernetesConfig(kubeconfig_path='/custom/kubeconfig')
    assert config_with_path.kubeconfig_path == '/custom/kubeconfig'

    # Test with None kubeconfig_path (default)
    config_default = KubernetesConfig()
    assert config_default.kubeconfig_path is None

    # Test with empty string
    config_empty = KubernetesConfig(kubeconfig_path='')
    assert config_empty.kubeconfig_path == ''
