#!/usr/bin/env bash
# Source this file to add the project to your shell PATH and PYTHONPATH
# Usage: source ./activate_pi_stack.sh

_PI_STACK_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project root to PATH so scripts in the repo can be executed directly
export PATH="${_PI_STACK_PROJECT_ROOT}:$PATH"

# Add project root to PYTHONPATH so `modules` and other imports resolve
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${_PI_STACK_PROJECT_ROOT}"
else
  case ":${PYTHONPATH}:" in
    *":${_PI_STACK_PROJECT_ROOT}:"*) ;;
    *) export PYTHONPATH="${_PI_STACK_PROJECT_ROOT}:$PYTHONPATH" ;;
  esac
fi

# Define convenient shell functions that invoke the repository scripts with
# the system Python. These forward all arguments.
pi-stack-generator() {
  python3 "${_PI_STACK_PROJECT_ROOT}/pi-stack-generator.py" "$@"
}

pi-hyperopt() {
  python3 "${_PI_STACK_PROJECT_ROOT}/hyperparameter-opt/hyperopt.py" "$@"
}

echo "pi-stack-generator and pi-hyperopt available (project root: ${_PI_STACK_PROJECT_ROOT})"
