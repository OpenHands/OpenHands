"""Alias setup functionality for OpenHands V1 CLI.

This module provides comprehensive alias setup functionality that addresses issue #10754
by recommending uv tool install as the preferred approach over uvx for better performance
and version control, while maintaining backward compatibility.
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.shortcuts import radiolist_dialog, yes_no_dialog


class AliasSetup:
    """Handles shell alias setup for OpenHands CLI."""
    
    # Shell profiles by shell type
    SHELL_PROFILES = {
        'bash': ['.bashrc', '.bash_profile', '.profile'],
        'zsh': ['.zshrc', '.zprofile', '.profile'],
        'fish': ['.config/fish/config.fish'],
        'powershell': ['Documents/PowerShell/Microsoft.PowerShell_profile.ps1',
                      'Documents/WindowsPowerShell/Microsoft.PowerShell_profile.ps1']
    }
    
    # Alias configurations
    ALIAS_CONFIGS = {
        'uv_tool': {
            'name': 'uv tool install (Recommended)',
            'description': 'Persistent installation with better performance and version control',
            'aliases': {
                'openhands': 'openhands',
                'oh': 'openhands'
            },
            'prerequisites': ['uv tool install --python 3.12 openhands'],
            'benefits': [
                'Better Performance: Pre-installed tool environment',
                'Version Control: Explicit update management with "uv tool upgrade openhands-ai"',
                'Reduced Complexity: No shell profile modification needed',
                'Faster Startup: Avoids temporary environment creation'
            ]
        },
        'uvx': {
            'name': 'uvx (Legacy)',
            'description': 'Temporary environment approach (current default)',
            'aliases': {
                'openhands': 'uvx --python 3.12 openhands',
                'oh': 'uvx --python 3.12 openhands'
            },
            'prerequisites': [],
            'benefits': [
                'No Installation: Works without persistent installation',
                'Always Fresh: Uses latest version from PyPI',
                'Isolation: Each run is in clean environment'
            ]
        }
    }

    def __init__(self):
        """Initialize alias setup."""
        self.home_dir = Path.home()
        self.shell_type = self._detect_shell()
        self.is_windows = platform.system() == 'Windows'
        
    def _detect_shell(self) -> str:
        """Detect the current shell type."""
        if platform.system() == 'Windows':
            return 'powershell'
        
        shell_path = os.environ.get('SHELL', '')
        if 'zsh' in shell_path:
            return 'zsh'
        elif 'fish' in shell_path:
            return 'fish'
        else:
            return 'bash'  # Default fallback
    
    def _check_uv_installed(self) -> bool:
        """Check if uv is installed and accessible."""
        try:
            result = subprocess.run(['uv', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_shell_profile_path(self) -> Optional[Path]:
        """Get the appropriate shell profile path for the current shell."""
        profiles = self.SHELL_PROFILES.get(self.shell_type, [])
        
        for profile in profiles:
            profile_path = self.home_dir / profile
            if profile_path.exists():
                return profile_path
        
        # Return the first (most common) profile path even if it doesn't exist
        if profiles:
            return self.home_dir / profiles[0]
        
        return None
    
    def _format_alias_command(self, alias_name: str, command: str) -> str:
        """Format alias command for the current shell."""
        if self.shell_type == 'fish':
            return f'alias {alias_name} "{command}"'
        elif self.shell_type == 'powershell':
            return f'function {alias_name} {{ {command} $args }}'
        else:  # bash/zsh
            return f'alias {alias_name}="{command}"'
    
    def _add_aliases_to_profile(self, profile_path: Path, aliases: Dict[str, str]) -> bool:
        """Add aliases to the shell profile."""
        try:
            # Create the directory if it doesn't exist (for fish config)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing content
            if profile_path.exists():
                content = profile_path.read_text(encoding='utf-8')
            else:
                content = ''
            
            # Prepare alias commands
            alias_commands = []
            alias_commands.append('# OpenHands CLI aliases')
            
            for alias_name, command in aliases.items():
                alias_cmd = self._format_alias_command(alias_name, command)
                alias_commands.append(alias_cmd)
            
            alias_block = '\n'.join(alias_commands) + '\n'
            
            # Check if aliases already exist
            if '# OpenHands CLI aliases' in content:
                return False  # Aliases already exist
            
            # Append aliases
            with profile_path.open('a', encoding='utf-8') as f:
                f.write('\n' + alias_block)
            
            return True
            
        except Exception as e:
            print_formatted_text(
                HTML(f'<red>Error writing to profile {profile_path}: {e}</red>')
            )
            return False
    
    def show_welcome_info(self):
        """Show welcome information about alias setup options."""
        print_formatted_text(HTML('<cyan><b>🚀 Welcome to OpenHands V1 CLI!</b></cyan>'))
        print_formatted_text(HTML(''))
        print_formatted_text(
            HTML('<yellow>Would you like to set up convenient shell aliases?</yellow>')
        )
        print_formatted_text(HTML(''))
        
        # Show information about different approaches
        uv_installed = self._check_uv_installed()
        
        if uv_installed:
            print_formatted_text(
                HTML('<green><b>💡 Recommended: uv tool install approach</b></green>')
            )
            print_formatted_text(
                HTML('<white>For better performance and version control, consider:</white>')
            )
            print_formatted_text(
                HTML('<cyan>  uv tool install --python 3.12 openhands-ai</cyan>')
            )
            print_formatted_text(
                HTML('<white>This provides persistent installation with explicit updates.</white>')
            )
            print_formatted_text(HTML(''))
        else:
            print_formatted_text(
                HTML('<yellow>⚠️  uv is not installed. Install from: </yellow><blue>https://docs.astral.sh/uv/getting-started/installation</blue>')
            )
            print_formatted_text(HTML(''))
    
    def run_alias_setup_flow(self) -> bool:
        """Run the complete alias setup flow."""
        self.show_welcome_info()
        
        # Ask user if they want to set up aliases
        setup_aliases = yes_no_dialog(
            title='Shell Alias Setup',
            text='Set up convenient shell aliases for OpenHands?'
        ).run()
        
        if not setup_aliases:
            print_formatted_text(HTML('<yellow>Skipping alias setup.</yellow>'))
            return False
        
        # Show approach selection
        uv_installed = self._check_uv_installed()
        approach_options = []
        
        if uv_installed:
            approach_options.append(('uv_tool', self.ALIAS_CONFIGS['uv_tool']['name']))
        
        approach_options.append(('uvx', self.ALIAS_CONFIGS['uvx']['name']))
        
        if len(approach_options) > 1:
            selected_approach = radiolist_dialog(
                title='Choose Installation Approach',
                text='Select your preferred approach:',
                values=approach_options
            ).run()
            
            if not selected_approach:
                print_formatted_text(HTML('<yellow>Alias setup cancelled.</yellow>'))
                return False
        else:
            selected_approach = approach_options[0][0]
        
        # Show detailed information about selected approach
        config = self.ALIAS_CONFIGS[selected_approach]
        print_formatted_text(HTML(''))
        print_formatted_text(
            HTML(f'<green><b>Selected: {config["name"]}</b></green>')
        )
        print_formatted_text(HTML(f'<white>{config["description"]}</white>'))
        print_formatted_text(HTML(''))
        
        # Show benefits
        print_formatted_text(HTML('<yellow><b>Benefits:</b></yellow>'))
        for benefit in config['benefits']:
            print_formatted_text(HTML(f'<white>  • {benefit}</white>'))
        print_formatted_text(HTML(''))
        
        # Handle prerequisites
        if config['prerequisites']:
            print_formatted_text(
                HTML('<yellow><b>Prerequisites (run these first):</b></yellow>')
            )
            for prereq in config['prerequisites']:
                print_formatted_text(HTML(f'<cyan>  {prereq}</cyan>'))
            print_formatted_text(HTML(''))
            
            proceed = yes_no_dialog(
                title='Prerequisites',
                text='Have you run the prerequisites above? Continue with alias setup?'
            ).run()
            
            if not proceed:
                print_formatted_text(
                    HTML('<yellow>Please run prerequisites first, then restart alias setup.</yellow>')
                )
                return False
        
        # Set up aliases
        return self._setup_aliases(config['aliases'], selected_approach)
    
    def _setup_aliases(self, aliases: Dict[str, str], approach: str) -> bool:
        """Set up the selected aliases."""
        profile_path = self._get_shell_profile_path()
        
        if not profile_path:
            print_formatted_text(
                HTML('<red>Could not determine shell profile location.</red>')
            )
            return False
        
        print_formatted_text(
            HTML(f'<white>Setting up aliases in: <cyan>{profile_path}</cyan></white>')
        )
        print_formatted_text(HTML(''))
        
        # Show what will be added
        print_formatted_text(HTML('<yellow><b>Aliases to be added:</b></yellow>'))
        for alias_name, command in aliases.items():
            print_formatted_text(
                HTML(f'<white>  • <cyan>{alias_name}</cyan> → <green>{command}</green></white>')
            )
        print_formatted_text(HTML(''))
        
        # Confirm
        confirm = yes_no_dialog(
            title='Confirm Alias Setup',
            text=f'Add these aliases to {profile_path.name}?'
        ).run()
        
        if not confirm:
            print_formatted_text(HTML('<yellow>Alias setup cancelled.</yellow>'))
            return False
        
        # Add aliases
        success = self._add_aliases_to_profile(profile_path, aliases)
        
        if success:
            print_formatted_text(
                HTML('<green><b>✅ Aliases added successfully!</b></green>')
            )
            print_formatted_text(
                HTML(f'<white>Run <cyan>source {profile_path}</cyan> or restart your terminal to use the new aliases.</white>')
            )
            
            # Show next steps for uv tool approach
            if approach == 'uv_tool':
                print_formatted_text(HTML(''))
                print_formatted_text(
                    HTML('<yellow><b>Next steps:</b></yellow>')
                )
                print_formatted_text(
                    HTML('<white>1. Run: <cyan>uv tool install --python 3.12 openhands-ai</cyan></white>')
                )
                print_formatted_text(
                    HTML('<white>2. Use: <cyan>openhands</cyan> or <cyan>oh</cyan> to start OpenHands</white>')
                )
                print_formatted_text(
                    HTML('<white>3. Update: <cyan>uv tool upgrade openhands-ai</cyan></white>')
                )
        else:
            print_formatted_text(
                HTML('<red>Failed to add aliases. They may already exist.</red>')
            )
        
        return success


def run_alias_setup() -> bool:
    """Run the alias setup flow.
    
    Returns:
        True if aliases were set up successfully, False otherwise
    """
    alias_setup = AliasSetup()
    return alias_setup.run_alias_setup_flow()
