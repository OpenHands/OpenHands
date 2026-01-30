# OpenHands Kubernetes Runtime

This directory contains the Kubernetes runtime implementation for OpenHands, which allows the software to run on Kubernetes clusters for scalable and isolated execution environments.

## Local Development with KIND

For local development and testing, OpenHands provides a convenient setup using KIND (Kubernetes IN Docker) that creates a local Kubernetes cluster.

### Prerequisites

Before setting up the local Kubernetes environment, ensure you have the following tools installed:

1. **KIND (Kubernetes IN Docker)** - [Installation Guide](https://kind.sigs.k8s.io/docs/user/quick-start/)

2. **kubectl** - [Installation Guide](https://kubernetes.io/docs/tasks/tools/#kubectl)

3. **mirrord** - [Installation Guide](https://metalbear.co/mirrord/docs/overview/quick-start/#installation)

   MirrorD is used for network mirroring allowing the locally running process to interact with the kubernetes cluster as if it were running inside of kubernetes.

4. **Docker or Podman** - Required for KIND to work
   - Docker: Follow the official Docker installation guide for your platform
   - Podman: [Installation Guide](https://podman.io/docs/installation)

### Configuration

To use the Kubernetes runtime, you need to configure OpenHands properly. The configuration is done through a TOML configuration file.

#### Required Configuration

Two configuration options are required to use the Kubernetes runtime:

1. **Runtime Type**: Set the runtime to use Kubernetes

   ```toml
   [core]
   runtime = "kubernetes"
   ```

2. **Runtime Container Image**: Specify the container image to use for the runtime environment
   ```toml
   [sandbox]
   runtime_container_image = "docker.openhands.dev/openhands/runtime:1.2-nikolaik"
   ```

#### Additional Kubernetes Options

OpenHands provides extensive configuration options for Kubernetes deployments under the `[kubernetes]` section. These options allow you to customize:

- Kubernetes namespace
- Persistent volume configuration
- Ingress and networking settings
- Runtime Pod Security settings
- Resource limits and requests

For a complete list of available Kubernetes configuration options, refer to the `[kubernetes]` section in the `config.template.toml` file in the repository root.

## Local Development Setup

### Quick Start

To set up and run OpenHands with the Kubernetes runtime locally:

First build the application with

```bash
make build
```

Then

```bash
make kind # target is stateless and will check for an existing kind cluster or make a new one if not present.
```

This command will:

1. **Check Dependencies**: Verify that `kind`, `kubectl`, and `mirrord` are installed
2. **Create KIND Cluster**: Create a local Kubernetes cluster named "local-hands" using the configuration in `kind/cluster.yaml`
3. **Deploy Infrastructure**: Apply Kubernetes manifests including:
   - Ubuntu development pod for runtime execution
   - Nginx ingress controller for HTTP routing
   - RBAC configurations for proper permissions
4. **Setup Mirrord**: Install mirrord resources for development workflow
5. **Run Application**: Execute `make run` inside the mirrord environment

### Cluster Configuration

The KIND cluster is configured with:

- **Cluster Name**: `local-hands`
- **Node Configuration**: Single control-plane node
- **Port Mapping**: Host port 80 maps to container port 80 for nginx ingress
- **Base Image**: Ubuntu 22.04 for the development environment

### Infrastructure Components

The local setup includes several Kubernetes resources:

#### Development Environment

- **Deployment**: `ubuntu-dev` - Ubuntu 22.04 container for code execution
- **Service**: Exposes the development environment within the cluster

#### Ingress Controller (Nginx)

- **Namespace**: `ingress-nginx` - Dedicated namespace for ingress resources
- **Deployment**: `ingress-nginx-controller` - Handles HTTP routing and load balancing
- **Service**: LoadBalancer service for external access
- **ConfigMap**: Custom configuration for nginx controller
- **RBAC**: Roles and bindings for proper cluster permissions

#### Development Workflow

- **Mirrord Integration**: Allows running local development server while connecting to cluster resources
- **Port Forwarding**: Direct access to cluster services from localhost

### Usage

Once the environment is set up with `make kind`, the system will:

1. Wait for all deployments to be ready
2. Automatically start the OpenHands application using mirrord
3. Provide access to the application at http://127.0.0.1:3000/

The mirrord integration allows you to develop locally while your application has access to the Kubernetes cluster resources, providing a seamless development experience that mirrors production behavior.

### Troubleshooting

If you encounter issues:

1. **Check cluster status**: `kubectl get nodes`
2. **Verify deployments**: `kubectl get deployments --all-namespaces`
3. **Check ingress**: `kubectl get ingress --all-namespaces`
4. **View logs**: `kubectl logs -l app=ubuntu-dev`

To clean up the environment:

```bash
kind delete cluster --name local-hands
```
