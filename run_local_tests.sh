#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
# Model-mock suites deliberately run in separate Python interpreters:
# PyTorch lazy imports can conflict with temporary fake-transformers modules.
PYTHON="${PYTHON_BIN:-}"
if [ -z "$PYTHON" ]; then
  if [ -x .venv_t5/bin/python ]; then PYTHON=.venv_t5/bin/python; else PYTHON=python3; fi
fi
"$PYTHON" -m compileall -q .
for suite in tests.test_static tests.test_core tests.test_bert_train_mock tests.test_t5_train_mock; do
  echo "=== $suite ==="
  "$PYTHON" -m unittest "$suite" -q
done
printf 'All offline regressions passed. Real pretrained/GPU metrics remain untested.\n'