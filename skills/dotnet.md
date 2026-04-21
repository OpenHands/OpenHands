---
name: dotnet
type: knowledge
version: 1.0.0
agent: CodeActAgent
triggers:
- dotnet
- .NET
- csproj
- nuget
- dotnet-sdk
- dotnet-install
---

# .NET SDK Installation Guide for Linux

The .NET SDK is not pre-installed in the OpenHands sandbox. When working on a C#/.NET
project (you'll typically see `*.csproj`, `*.sln`, `*.fsproj`, or `global.json` files),
install the SDK yourself before building or running tests.

Do **not** rely on a project-provided `install.sh`; those scripts usually assume
elevated privileges and a particular apt repo layout and will frequently fail in the
sandbox. Use one of the two approaches below instead.

## Preferred: install into the user home with `dotnet-install.sh`

This is the most reliable approach in the sandbox because it does not require `sudo`
and does not touch system directories. It is the method recommended by Microsoft for
non-root installs.

1. Check whether the project pins a specific SDK version in `global.json`. If it does,
   install that exact version. Otherwise install the latest LTS (`--channel LTS`) or
   a specific channel (e.g. `--channel 10.0`, `--channel 9.0`, `--channel 8.0`).

2. Download and run the official install script:

   ```bash
   # Install latest LTS (currently .NET 10) into ~/.dotnet
   curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
   chmod +x /tmp/dotnet-install.sh
   /tmp/dotnet-install.sh --channel LTS --install-dir "$HOME/.dotnet"
   ```

   To pin a specific channel or version:

   ```bash
   # Specific channel (major.minor)
   /tmp/dotnet-install.sh --channel 10.0 --install-dir "$HOME/.dotnet"

   # Exact version from global.json
   /tmp/dotnet-install.sh --version 10.0.100 --install-dir "$HOME/.dotnet"
   ```

3. Put `dotnet` on `PATH` for the current shell and future shells:

   ```bash
   export DOTNET_ROOT="$HOME/.dotnet"
   export PATH="$DOTNET_ROOT:$PATH"
   echo 'export DOTNET_ROOT="$HOME/.dotnet"' >> ~/.bashrc
   echo 'export PATH="$DOTNET_ROOT:$PATH"' >> ~/.bashrc
   ```

4. Suppress the first-run telemetry prompt so it doesn't interfere with non-interactive
   commands:

   ```bash
   export DOTNET_CLI_TELEMETRY_OPTOUT=1
   export DOTNET_NOLOGO=1
   ```

5. Verify the installation:

   ```bash
   dotnet --info
   dotnet --list-sdks
   ```

## Alternative: install via apt (requires sudo)

The sandbox's `openhands` user has passwordless sudo, so the Microsoft apt package
feed also works. Use this only if you specifically need the system-wide install
location (`/usr/share/dotnet`):

```bash
# Ubuntu 24.04 / 22.04 — adjust the version in the URL for other distros
source /etc/os-release
wget -q "https://packages.microsoft.com/config/${ID}/${VERSION_ID}/packages-microsoft-prod.deb" -O /tmp/packages-microsoft-prod.deb
sudo dpkg -i /tmp/packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install -y dotnet-sdk-10.0
```

If `packages-microsoft-prod.deb` is not available for the detected distro (e.g.
Debian unstable), fall back to the `dotnet-install.sh` method above.

## Building and testing a project

Once the SDK is on `PATH`:

```bash
dotnet restore
dotnet build --no-restore
dotnet test --no-build --logger "console;verbosity=normal"
```

If `dotnet restore` is slow or fails due to network restrictions, check whether the
project has a `NuGet.config` that points at a private feed; you may need credentials
from the user before the restore can succeed.
