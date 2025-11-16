# run_experiments.py
#!/usr/bin/env python3
import argparse

from rag.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple RAG experiments with different configs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Base config.yaml",
    )
    parser.add_argument(
        "--experiments-yaml",
        type=str,
        default="experiments.yaml",
        help="YAML с описанием экспериментов",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Eval dataset path (JSONL). Если не задан, берётся из config.eval.dataset_path",
    )
    args = parser.parse_args()

    run_experiments(
        config_path=args.config,
        experiments_yaml=args.experiments_yaml,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    main()