#!/usr/bin/env python3
"""
Batch experiment runner for cross-lingual IR experiments.

Orchestrates complete end-to-end pipelines:
- Index building (BM25, dense)
- Retrieval (BM25, mDPR, ColBERT)
- Reranking (monoT5/mT5)
- Evaluation (trec_eval)

Usage:
    python scripts/run_experiments.py --config config/neuclir.yaml --pipeline bm25
    python scripts/run_experiments.py --config config/neuclir.yaml --pipeline dense_mdpr
    python scripts/run_experiments.py --config config/neuclir.yaml --pipeline full
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List
import time

from utils_io import load_yaml, get_repo_root, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates batch experiments."""

    def __init__(self, config: Dict[str, Any], repo_root: Path):
        """
        Initialize experiment runner.

        Args:
            config: Configuration dictionary
            repo_root: Repository root path
        """
        self.config = config
        self.repo_root = repo_root
        self.scripts_dir = repo_root / "scripts"

    def run_command(self, cmd: List[str], description: str) -> bool:
        """
        Run a command and log output.

        Args:
            cmd: Command to run
            description: Description for logging

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"{'='*80}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True
            )
            logger.info(f"✓ {description} completed successfully\n")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {description} failed with return code {e.returncode}\n")
            return False
        except FileNotFoundError as e:
            logger.error(f"✗ Command not found: {e}\n")
            return False

    def run_bm25_pipeline(self, languages: List[str]) -> None:
        """
        Run complete BM25 pipeline.

        Args:
            languages: List of language codes
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING BM25 PIPELINE")
        logger.info("="*80 + "\n")

        for lang in languages:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Processing language: {lang}")
            logger.info(f"{'#'*80}\n")

            # Build index
            cmd = [
                sys.executable,
                str(self.scripts_dir / "build_index_bm25.py"),
                "--config", "config/neuclir.yaml",
                "--lang", lang
            ]
            if not self.run_command(cmd, f"Build BM25 index for {lang}"):
                continue

            # Run retrieval
            cmd = [
                sys.executable,
                str(self.scripts_dir / "run_bm25.py"),
                "--config", "config/neuclir.yaml",
                "--lang", lang
            ]
            if not self.run_command(cmd, f"Run BM25 retrieval for {lang}"):
                continue

            # Evaluate
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate.py"),
                "--config", "config/neuclir.yaml",
                "--run_dir", "runs/bm25",
                "--lang", lang
            ]
            self.run_command(cmd, f"Evaluate BM25 results for {lang}")

    def run_dense_mdpr_pipeline(self, languages: List[str]) -> None:
        """
        Run complete mDPR dense retrieval pipeline.

        Args:
            languages: List of language codes
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING mDPR DENSE PIPELINE")
        logger.info("="*80 + "\n")

        for lang in languages:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Processing language: {lang}")
            logger.info(f"{'#'*80}\n")

            # Build index
            cmd = [
                sys.executable,
                str(self.scripts_dir / "build_index_dense.py"),
                "--config", "config/neuclir.yaml",
                "--model", "mdpr",
                "--lang", lang
            ]
            if not self.run_command(cmd, f"Build mDPR index for {lang}"):
                continue

            # Run retrieval
            cmd = [
                sys.executable,
                str(self.scripts_dir / "run_dense_mdpr.py"),
                "--config", "config/neuclir.yaml",
                "--lang", lang
            ]
            if not self.run_command(cmd, f"Run mDPR retrieval for {lang}"):
                continue

            # Evaluate
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate.py"),
                "--config", "config/neuclir.yaml",
                "--run_dir", "runs/dense",
                "--lang", lang
            ]
            self.run_command(cmd, f"Evaluate mDPR results for {lang}")

    def run_reranking_pipeline(self, languages: List[str], base_run_dir: str) -> None:
        """
        Run reranking pipeline on existing runs.

        Args:
            languages: List of language codes
            base_run_dir: Directory containing base runs (e.g., 'runs/bm25' or 'runs/dense')
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING RERANKING PIPELINE")
        logger.info("="*80 + "\n")

        base_run_path = self.repo_root / base_run_dir

        for lang in languages:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# Processing language: {lang}")
            logger.info(f"{'#'*80}\n")

            # Find base run files for this language
            run_files = list(base_run_path.glob(f"*{lang}*.run"))

            if not run_files:
                logger.warning(f"No run files found for {lang} in {base_run_path}")
                continue

            for run_file in run_files:
                logger.info(f"\nReranking: {run_file.name}")

                # Rerank
                cmd = [
                    sys.executable,
                    str(self.scripts_dir / "rerank_mt5.py"),
                    "--config", "config/neuclir.yaml",
                    "--base_run", str(run_file),
                    "--lang", lang
                ]
                if not self.run_command(cmd, f"Rerank {run_file.name}"):
                    continue

            # Evaluate reranked results
            cmd = [
                sys.executable,
                str(self.scripts_dir / "evaluate.py"),
                "--config", "config/neuclir.yaml",
                "--run_dir", "runs/reranked",
                "--lang", lang
            ]
            self.run_command(cmd, f"Evaluate reranked results for {lang}")

    def run_full_pipeline(self, languages: List[str]) -> None:
        """
        Run complete end-to-end pipeline.

        Args:
            languages: List of language codes
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING FULL END-TO-END PIPELINE")
        logger.info("="*80 + "\n")

        # Run BM25 pipeline
        self.run_bm25_pipeline(languages)

        # Run dense pipeline
        self.run_dense_mdpr_pipeline(languages)

        # Run reranking on BM25 results
        self.run_reranking_pipeline(languages, "runs/bm25")

        # Run reranking on dense results
        self.run_reranking_pipeline(languages, "runs/dense")

        logger.info("\n" + "="*80)
        logger.info("FULL PIPELINE COMPLETE!")
        logger.info("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for cross-lingual IR"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        choices=['bm25', 'dense_mdpr', 'rerank', 'full'],
        required=True,
        help='Pipeline to run'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=None,
        help='Languages to process (default: all from config)'
    )
    parser.add_argument(
        '--base_run_dir',
        type=str,
        default='runs/bm25',
        help='Base run directory for reranking (default: runs/bm25)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_yaml(args.config)
    repo_root = get_repo_root()

    # Get languages
    if args.languages:
        languages = args.languages
    else:
        languages = config['languages']

    logger.info(f"Languages to process: {languages}")

    # Initialize runner
    runner = ExperimentRunner(config, repo_root)

    # Record start time
    start_time = time.time()

    # Run specified pipeline
    if args.pipeline == 'bm25':
        runner.run_bm25_pipeline(languages)
    elif args.pipeline == 'dense_mdpr':
        runner.run_dense_mdpr_pipeline(languages)
    elif args.pipeline == 'rerank':
        runner.run_reranking_pipeline(languages, args.base_run_dir)
    elif args.pipeline == 'full':
        runner.run_full_pipeline(languages)

    # Record end time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logger.info(f"\nTotal time: {hours}h {minutes}m {seconds}s")
    logger.info("All experiments complete!")


if __name__ == '__main__':
    main()
