# Offline validation — A4 V6

Executed in this container on 2026-11-17. No NYU Torch GPU or genuine model checkpoints were available here.

Commands:

```bash
bash run_local_tests.sh
```

Actual outcome: **14/14 offline unit/regression tests passed**, run as 4 isolated suites. Static checks include Python AST syntax, all expected paths, CLI `--help` compatibility, Slurm entrypoint wiring. Integration checks cover stage failure isolation, content/signature validation for stale BERT predictions, deterministic OOD transform, SQLite SQL/Record F1 evaluation, tuple PKL format, actual PyTorch gradients/Optimizer state checkpoint saving, Q3 training interruption and resume, T5 training interruption and resume (2 mock epochs), and Test SQL generation interruption at 24/432 queries followed by recovery to 432 SQL plus 432 record sets.

Actual model weights/tokenizers were replaced with small fake implementations for training/inference tests. The tests verify code execution, state preservation, and output format, **not** actual pretrained BERT/T5 convergence or hidden Test Record F1. `preflight.py` will perform real one-example CPU forward passes on the NYU login node using the existing scratch environments, pretrained weights and official data before GPU submission.

Known validation gaps before a real HPC run: cannot verify Q2 >4 pp or Test Record F1 >=65, cannot guarantee Slurm queue/preemptible policy, and cannot measure full training duration. When real data or model dependencies fail preflight, fix those errors before requesting a GPU.