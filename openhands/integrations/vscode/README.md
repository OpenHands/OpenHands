# OpenHands VS Code Extension

The official OpenHands companion extension for Visual Studio Code.

This extension seamlessly integrates OpenHands into your VSCode workflow, allowing you to start coding sessions with your AI agent directly from your editor.

![OpenHands VSCode Extension Demo](https://raw.githubusercontent.com/OpenHands/OpenHands/main/assets/images/vscode-extension-demo.gif)

## Features

- **Start a New Conversation**: Launch OpenHands in a new terminal with a single command.
- **Use Your Current File**: Automatically send the content of your active file to OpenHands to start a task.
- **Use a Selection**: Send only the highlighted text from your editor to OpenHands for focused tasks.
- **Safe Terminal Management**: The extension intelligently reuses idle terminals or creates new ones, ensuring it never interrupts an active process.
- **Automatic Virtual Environment Detection**: Finds and uses your project's Python virtual environment (`.venv`, `venv`, etc.) automatically.

## How to Use

You can access the extension's commands in two ways:

1.  **Command Palette**:
    - Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`).
    - Type `OpenHands` to see the available commands.
    - Select the command you want to run.

2.  **Editor Context Menu**:
    - Right-click anywhere in your text editor.
    - The OpenHands commands will appear in the context menu.

## Installation

For the best experience, the OpenHands CLI will attempt to install the extension for you automatically the first time you run it inside VSCode.

If you need to install it manually:
1.  Download the latest `.vsix` file from the [GitHub Releases page](https://github.com/OpenHands/OpenHands/releases).
2.  In VSCode, open the Command Palette (`Ctrl+Shift+P`).
3.  Run the **"Extensions: Install from VSIX..."** command.
4.  Select the `.vsix` file you downloaded.

## Requirements

- **OpenHands CLI**: You must have `openhands` installed and available in your system's PATH.
- **VS Code**: Version 1.98.2 or newer.
- **Shell**: For the best terminal reuse experience, a shell with [Shell Integration](https://code.visualstudio.com/docs/terminal/shell-integration) is recommended (e.g., modern versions of bash, zsh, PowerShell, or fish).

## Contributing

We welcome contributions! If you're interested in developing the extension, please see the `DEVELOPMENT.md` file in our source repository for instructions on how to get started.
