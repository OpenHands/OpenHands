#!/bin/bash

echo "Running OpenHands pre-commit hook..."
echo "This hook runs selective linting based on changed files."

# Store the exit code to return at the end
# This allows us to be additive to existing pre-commit hooks
EXIT_CODE=0

# Get the list of staged files
STAGED_FILES=$(git diff --cached --name-only)

# Check if any files match specific patterns
has_frontend_changes=false
has_backend_changes=false
has_vscode_changes=false

# Check each file individually to avoid issues with grep
for file in $STAGED_FILES; do
    if [[ $file == frontend/* ]]; then
        has_frontend_changes=true
    elif [[ $file == openhands/* || $file == evaluation/* || $file == tests/* ]]; then
        has_backend_changes=true
        # Check for VSCode extension changes (subset of backend changes)
        if [[ $file == openhands/integrations/vscode/* ]]; then
            has_vscode_changes=true
        fi
    fi
done

echo "Analyzing changes..."
echo "- Frontend changes: $has_frontend_changes"
echo "- Backend changes: $has_backend_changes"
echo "- VSCode extension changes: $has_vscode_changes"

# Run frontend linting if needed
if [ "$has_frontend_changes" = true ]; then
    # Check if we're in a CI environment or if frontend dependencies are missing
    if [ -n "$CI" ] || ! command -v react-router &> /dev/null || ! command -v vitest &> /dev/null; then
        echo "Skipping frontend checks (CI environment or missing dependencies detected)."
        echo "WARNING: Frontend files have changed but frontend checks are being skipped."
        echo "Please run 'make lint-frontend' manually before submitting your PR."
    else
        echo "Running frontend linting..."
        make lint-frontend
        if [ $? -ne 0 ]; then
            echo "Frontend linting failed. Please fix the issues before committing."
            EXIT_CODE=1
        else
            echo "Frontend linting checks passed!"
        fi

        # Run additional frontend checks
        if [ -d "frontend" ]; then
            echo "Running additional frontend checks..."
            cd frontend || exit 1

            # Run build
            echo "Running npm build..."
            npm run build
            if [ $? -ne 0 ]; then
                echo "Frontend build failed. Please fix the issues before committing."
                EXIT_CODE=1
            fi

            # Run tests
            echo "Running npm test..."
            npm test
            if [ $? -ne 0 ]; then
                echo "Frontend tests failed. Please fix the failing tests before committing."
                EXIT_CODE=1
            fi

            cd ..
        fi
    fi
else
    echo "Skipping frontend checks (no frontend changes detected)."
fi

# Run backend linting if needed
if [ "$has_backend_changes" = true ]; then
    echo "Running backend linting..."
    make lint-backend
    if [ $? -ne 0 ]; then
        echo "Backend linting failed. Please fix the issues before committing."
        EXIT_CODE=1
    else
        echo "Backend linting checks passed!"
    fi
else
    echo "Skipping backend checks (no backend changes detected)."
fi

# Run VSCode extension checks if needed
if [ "$has_vscode_changes" = true ]; then
    # Check if we're in a CI environment
    if [ -n "$CI" ]; then
        echo "Skipping VSCode extension checks (CI environment detected)."
        echo "WARNING: VSCode extension files have changed but checks are being skipped."
        echo "Please run VSCode extension checks manually before submitting your PR."
    else
        echo "Running VSCode extension checks..."
        if [ -d "openhands/integrations/vscode" ]; then
            cd openhands/integrations/vscode || exit 1

            echo "Running npm lint:fix..."
            npm run lint:fix
            if [ $? -ne 0 ]; then
                echo "VSCode extension linting failed. Please fix the issues before committing."
                EXIT_CODE=1
            else
                echo "VSCode extension linting passed!"
            fi

            echo "Running npm typecheck..."
            npm run typecheck
            if [ $? -ne 0 ]; then
                echo "VSCode extension type checking failed. Please fix the issues before committing."
                EXIT_CODE=1
            else
                echo "VSCode extension type checking passed!"
            fi

            echo "Running npm compile..."
            npm run compile
            if [ $? -ne 0 ]; then
                echo "VSCode extension compilation failed. Please fix the issues before committing."
                EXIT_CODE=1
            else
                echo "VSCode extension compilation passed!"
            fi

            cd ../../..
        fi
    fi
else
    echo "Skipping VSCode extension checks (no VSCode extension changes detected)."
fi

# If no specific code changes detected, run basic checks
if [ "$has_frontend_changes" = false ] && [ "$has_backend_changes" = false ]; then
    echo "No specific code changes detected. Running basic checks..."
    if [ -n "$STAGED_FILES" ]; then
        # Run only basic pre-commit hooks for non-code files
        poetry run pre-commit run --files $(echo "$STAGED_FILES" | tr '\n' ' ') --hook-stage commit --config ./dev_config/python/.pre-commit-config.yaml
        if [ $? -ne 0 ]; then
            echo "Basic checks failed. Please fix the issues before committing."
            EXIT_CODE=1
        else
            echo "Basic checks passed!"
        fi
    else
        echo "No files changed. Skipping basic checks."
    fi
fi

# Run any existing pre-commit hooks that might have been installed by the user
# This makes our hook additive rather than replacing existing hooks
if [ -f ".git/hooks/pre-commit.local" ]; then
    echo "Running existing pre-commit hooks..."
    bash .git/hooks/pre-commit.local
    if [ $? -ne 0 ]; then
        echo "Existing pre-commit hooks failed."
        EXIT_CODE=1
    fi
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "All pre-commit checks passed!"
else
    echo "Some pre-commit checks failed. Please fix the issues before committing."
fi

exit $EXIT_CODE
