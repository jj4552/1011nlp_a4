# DS-GA 1011 Assignment 4 (Fall 2026)

Source code for BERT sentiment classification (Q1-Q3) and T5-small text-to-SQL (Q4-Q7).

## Structure
- `part1/main.py`: fine-tuning BERT on IMDB, clean/transformed evaluation, data augmentation.
- `part1/utils.py`: deterministic keyboard typo transformation for Q2 and Q3.
- `part2/t5_engine.py`: fine-tuning T5-small on ATIS and generating dev/test SQL.
- `part2/sql_eval.py`: SQLite record-based evaluation.
- `part2/q4_q6.py`: token statistics and dev error analysis.
- `run_pipeline.py`, `run_a4.sbatch`: resumable NYU HPC execution.

## Measured results
- Q1 clean accuracy: 92.572%.
- Q2 transformed accuracy: 86.624% (5.948 percentage-point decrease).
- Q3 augmented clean accuracy: 92.760%.
- Q3 augmented transformed accuracy: 88.384%.
- T5 dev Record F1: 80.031%; Record EM: 75.322%; SQL EM: 2.361%.

Held-out test F1 is assessed by Gradescope. Large model checkpoints and course datasets are not stored in the code repository; the PDF report provides the checkpoint folder link.

## Reproduction
Prepare the official course datasets as directed by the assignment, then use `setup_data.py` and `preflight.py` to check the environment. On a NYU Slurm GPU allocation use `sbatch run_a4.sbatch`. Some scripts require the official ATIS SQLite database, dataset files, and Hugging Face dependencies.
