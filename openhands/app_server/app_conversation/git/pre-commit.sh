#!/bin/bash
# This hook was installed by OpenHands
# It calls the pre-commit script in the .openhands directory

if [ -x ".openhands/pre-commit.sh" ]; then
    source ".openhands/pre-commit.sh"
    exit $?
else
    echo "Warning: .openhands/pre-commit.sh not found or not executable"
    exit 0
fi
