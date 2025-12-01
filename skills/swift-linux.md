---
 name: swift-linux
 type: knowledge
 agent: CodeActAgent
 version: 1.0.0
 triggers:
 - swift-linux
 - swift-debian
 - swift-installation
---

# Swift Installation Guide for Debian Linux

This document provides instructions for installing Swift on Debian 12 (Bookworm).

> This setup is intended for non-UI development tasks on Swift on Linux.

## Prerequisites

Before installing Swift, you need to install the required dependencies for your system. You can find the most up-to-date list of dependencies for your specific Linux distribution and version at the [Swift.org tarball installation guide](https://www.swift.org/install/linux/tarball/).

FOR EXAMPLE, the dependencies you may need to install for Debian 12 could be:

```bash
sudo apt-get update
sudo apt-get install -y \
  binutils-gold \
  gcc \
  git \
  libcurl4-openssl-dev \
  libedit-dev \
  libicu-dev \
  libncurses-dev \
  libpython3-dev \
  libsqlite3-dev \
  libxml2-dev \
  pkg-config \
  tzdata \
  uuid-dev
```

## Download and Install Swift

1. Find the latest Swift version for Debian:

   Go to the [Swift.org download page](https://www.swift.org/download/) to find the latest Swift version compatible with Debian 12 (Bookworm).

   Look for a tarball named something like `swift-<VERSION>-RELEASE-debian12.tar.gz` (e.g., `swift-6.0.3-RELEASE-debian12.tar.gz`).

   The URL pattern is typically:
   ```
   https://download.swift.org/swift-<VERSION>-release/debian12/swift-<VERSION>-RELEASE/swift-<VERSION>-RELEASE-debian12.tar.gz
   ```

   Where `<VERSION>` is the Swift version number (e.g., `6.0.3`).

2. Download the Swift binary for Debian 12:

```bash
cd /workspace
wget https://download.swift.org/swift-6.0.3-release/debian12/swift-6.0.3-RELEASE/swift-6.0.3-RELEASE-debian12.tar.gz
```

3. Extract the archive:

> **Note**: Make sure to install Swift in the `/workspace` directory, but outside the git repository to avoid committing the Swift binaries.

4. Add Swift to your PATH by adding the following line to your `~/.bashrc` file:

```bash
echo 'export PATH=/workspace/swift-6.0.3-RELEASE-debian12/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

> **Note**: Make sure to update the version number in the PATH to match the version you downloaded.

## Verify Installation

Verify that Swift is correctly installed by running:

```bash
swift --version
```
