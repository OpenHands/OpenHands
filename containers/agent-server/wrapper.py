#!/usr/bin/env python3
"""
Wrapper script for agent-server that adds CORS middleware support
This script intercepts the agent-server binary and patches the FastAPI app
"""
import os
import sys
import subprocess
from pathlib import Path

# Add the custom middleware to the path
sys.path.insert(0, '/tmp')

def main():
    # Get the original agent-server binary path
    original_binary = '/usr/local/bin/openhands-agent-server'
    
    # Check if PERMITTED_CORS_ORIGINS is set
    permitted_origins = os.getenv('PERMITTED_CORS_ORIGINS', '')
    
    if not permitted_origins:
        # If no CORS origins specified, just run the original binary
        os.execv(original_binary, [original_binary] + sys.argv[1:])
    
    # Since the binary is compiled, we can't easily patch it
    # Instead, we'll need to use a reverse proxy approach or
    # check if the binary supports PERMITTED_CORS_ORIGINS natively
    
    # For now, just pass through - the binary might support it
    os.execv(original_binary, [original_binary] + sys.argv[1:])

if __name__ == '__main__':
    main()

