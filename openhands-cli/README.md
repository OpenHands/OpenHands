# OpenHands V1 CLI

A **lightweight, modern CLI** to interact with the OpenHands agent (powered by [OpenHands software-agent-sdk](https://github.com/OpenHands/software-agent-sdk)). 

---

## Quickstart

- Prerequisites: Python 3.12+, curl
- Install uv (package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Restart your shell so "uv" is on PATH, or follow the installer hint
  ```

### Run the CLI locally
```bash
make install

# Start the CLI
make run
# or
uv run openhands
```

### Build a standalone executable
```bash
# Build (installs PyInstaller if needed)
./build.sh --install-pyinstaller

# The binary will be in dist/
./dist/openhands            # macOS/Linux
# dist/openhands.exe        # Windows
```

---

## Troubleshooting

### "Prompt not visible" or RuntimeError on startup

If you encounter errors like `RuntimeError: Prompt not visible after init`, this means your terminal environment doesn't support interactive prompts.

**Common causes:**
- Running in a non-interactive shell (piped input)
- TERM environment variable not set or set to 'dumb'
- Running in Docker without `-it` flags
- SSH connection without PTY allocation (`-t` flag)
- Using tmux/screen with broken terminal settings

**Solutions:**

1. **Check if you have a proper terminal:**
   ```bash
   # Check if TTY is available
   tty
   # Should output something like: /dev/pts/0
   
   # Check TERM variable
   echo $TERM
   # Should output something like: xterm-256color
   ```

2. **Set proper TERM variable:**
   ```bash
   export TERM=xterm-256color
   # Then try running openhands again
   ```

3. **Use TTY with Docker:**
   ```bash
   # Wrong (no TTY)
   docker run openhands/cli openhands
   
   # Correct (with TTY)
   docker run -it openhands/cli openhands
   ```

4. **Use PTY with SSH:**
   ```bash
   # Wrong (no PTY)
   ssh user@host "openhands"
   
   # Correct (with PTY)
   ssh -t user@host "openhands"
   ```

5. **Check tmux/screen settings:**
   ```bash
   # Inside tmux, check TERM
   echo $TERM
   # Should be: screen-256color or tmux-256color
   
   # If not, update your tmux.conf:
   set -g default-terminal "screen-256color"
   ```

6. **Install terminal info database (Linux):**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ncurses-term
   
   # RHEL/CentOS
   sudo yum install ncurses-term
   ```

### Running in CI or automation

- Interactive prompts are still required for human-driven sessions. In automated pipelines, set `OPENHANDS_CLI_SKIP_TTY_CHECK=1` (or ensure your CI exports `CI=true`) to bypass the TTY guard intentionally.
- The CLI will print a notice when the guard is skipped. Only use this override for trusted automation where interactive prompts are not expected.

### Other common issues

#### "uvx: command not found"
- Solution: Restart your terminal after installing uv, or add to PATH:
  ```bash
  export PATH="$HOME/.local/bin:$PATH"
  ```

#### "Docker is not installed"
- This error appears when running `openhands serve` (GUI mode)
- Solution: Install Docker from https://docs.docker.com/get-docker/
- Note: CLI mode (`openhands`) doesn't require Docker

#### "Docker daemon is not running"
- Solution: Start Docker Desktop or Docker service
- Check status: `docker info`

For more help, visit the [OpenHands documentation](https://docs.all-hands.dev/usage/troubleshooting).
