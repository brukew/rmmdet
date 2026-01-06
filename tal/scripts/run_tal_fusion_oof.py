#!/usr/bin/env python3
"""
Generate fused TAL-format window predictions for V-JEPA + skeleton model pairs.

This script runs OOF (out-of-fold) fusion for:
1. V-JEPA + PoseC3D (CE)
2. V-JEPA + STGCN++ 4-stream (CE)

For each pair and each CV fold, it:
- Loads val window predictions from both modalities
- Trains a fusion MLP via grouped inner-CV
- Outputs a fused TAL-format CSV ready for postprocessing grid search

Usage:
    python scripts/run_tal_fusion_oof.py
    
    # With custom config
    python scripts/run_tal_fusion_oof.py --hidden-dim 24 --num-epochs 150
    
    # Single pair only
    python scripts/run_tal_fusion_oof.py --pairs vjepa_posec3d
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tal_fusion import FusionConfig, run_oof_fusion, save_fused_predictions

logger = logging.getLogger(__name__)


# =============================================================================
# Path Configuration
# =============================================================================

# Root directories
TAL_DIR = Path("/orcd/data/satra/001/users/brukew/actreg/tal")
DATAPREP_DIR = Path("/orcd/data/satra/001/users/brukew/actreg/dataprep/tal")
EVAL_RESULTS_DIR = TAL_DIR / "eval_results"

# Window splits (val windows with labels)
WINDOW_SPLITS_PATTERN = DATAPREP_DIR / "splits_cv_4class" / "fold_{fold}_val_windows.csv"

# Existing TAL-format prediction paths
PRED_PATHS = {
    "vjepa": {
        "pattern": EVAL_RESULTS_DIR / "vjepa" / "fold{fold}" / "tal_format_preds.csv",
        "folds": [0, 1],
    },
    "posec3d_ce": {
        "pattern": EVAL_RESULTS_DIR / "posec3d_ce" / "fold{fold}" / "tal_format_preds.csv",
        "folds": [0, 1],
    },
    "stgcnpp_ce_4stream": {
        "pattern": EVAL_RESULTS_DIR / "stgcnpp_ce_4stream" / "fold{fold}" / "fused_tal_format_preds.csv",
        "folds": [0, 1],
    },
}

# Fusion pairs to generate
FUSION_PAIRS = {
    "vjepa_posec3d": {
        "mod1": "vjepa",
        "mod2": "posec3d_ce",
        "output_name": "vjepa_posec3d_mlp_logp",
        "description": "V-JEPA + PoseC3D (CE)",
    },
    "vjepa_stgcnpp": {
        "mod1": "vjepa",
        "mod2": "stgcnpp_ce_4stream",
        "output_name": "vjepa_stgcnpp_mlp_logp",
        "description": "V-JEPA + STGCN++ 4-stream (CE)",
    },
}


# =============================================================================
# Main Logic
# =============================================================================

def get_available_folds(pair_key: str) -> List[int]:
    """
    Get folds available for both modalities in a pair.
    
    Args:
        pair_key: Key from FUSION_PAIRS
    
    Returns:
        List of fold indices available for both modalities
    """
    pair_config = FUSION_PAIRS[pair_key]
    mod1_folds = set(PRED_PATHS[pair_config["mod1"]]["folds"])
    mod2_folds = set(PRED_PATHS[pair_config["mod2"]]["folds"])
    
    return sorted(mod1_folds & mod2_folds)


def run_fusion_for_pair(
    pair_key: str,
    folds: Optional[List[int]] = None,
    config: Optional[FusionConfig] = None,
    out_root: Optional[Path] = None,
) -> Dict:
    """
    Run OOF fusion for a modality pair across all available folds.
    
    Args:
        pair_key: Key from FUSION_PAIRS (e.g., "vjepa_posec3d")
        folds: List of folds to process (default: all available)
        config: Fusion configuration
        out_root: Output root directory
    
    Returns:
        Summary dict with per-fold statistics
    """
    if config is None:
        config = FusionConfig()
    
    if out_root is None:
        out_root = EVAL_RESULTS_DIR
    
    pair_config = FUSION_PAIRS[pair_key]
    mod1_key = pair_config["mod1"]
    mod2_key = pair_config["mod2"]
    output_name = pair_config["output_name"]
    
    # Determine folds
    if folds is None:
        folds = get_available_folds(pair_key)
    
    logger.info("=" * 70)
    logger.info("Fusion: %s", pair_config["description"])
    logger.info("=" * 70)
    logger.info("Modality 1: %s", mod1_key)
    logger.info("Modality 2: %s", mod2_key)
    logger.info("Output: %s", output_name)
    logger.info("Folds: %s", folds)
    
    # Process each fold
    fold_stats = {}
    
    for fold_idx in folds:
        logger.info("\n" + "-" * 50)
        logger.info("Processing fold %d", fold_idx)
        logger.info("-" * 50)
        
        # Build paths
        window_splits_csv = Path(str(WINDOW_SPLITS_PATTERN).format(fold=fold_idx))
        mod1_csv = Path(str(PRED_PATHS[mod1_key]["pattern"]).format(fold=fold_idx))
        mod2_csv = Path(str(PRED_PATHS[mod2_key]["pattern"]).format(fold=fold_idx))
        out_dir = out_root / output_name / f"fold{fold_idx}"
        
        # Validate paths exist
        if not window_splits_csv.exists():
            logger.warning("  Skipping fold %d: window splits not found at %s", fold_idx, window_splits_csv)
            continue
        if not mod1_csv.exists():
            logger.warning("  Skipping fold %d: mod1 preds not found at %s", fold_idx, mod1_csv)
            continue
        if not mod2_csv.exists():
            logger.warning("  Skipping fold %d: mod2 preds not found at %s", fold_idx, mod2_csv)
            continue
        
        # Run fusion
        try:
            output_df, stats = run_oof_fusion(
                window_splits_csv=window_splits_csv,
                modality1_csv=mod1_csv,
                modality2_csv=mod2_csv,
                config=config,
            )
            
            # Save outputs
            save_fused_predictions(output_df, stats, out_dir)
            
            fold_stats[fold_idx] = stats
            logger.info("  Fold %d complete: OOF accuracy = %.4f", fold_idx, stats["oof_accuracy"])
            
        except Exception as e:
            logger.error("  Fold %d failed: %s", fold_idx, e)
            fold_stats[fold_idx] = {"error": str(e)}
    
    # Aggregate stats
    summary = {
        "pair": pair_key,
        "description": pair_config["description"],
        "output_dir": str(out_root / output_name),
        "folds_processed": list(fold_stats.keys()),
        "per_fold": fold_stats,
    }
    
    # Compute mean OOF accuracy
    oof_accs = [s["oof_accuracy"] for s in fold_stats.values() if "oof_accuracy" in s]
    if oof_accs:
        summary["mean_oof_accuracy"] = float(sum(oof_accs) / len(oof_accs))
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate fused TAL predictions for V-JEPA + skeleton pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Pair selection
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=list(FUSION_PAIRS.keys()) + ["all"],
        default=["all"],
        help="Fusion pairs to run (default: all).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific folds to process (default: all available).",
    )
    
    # Output
    parser.add_argument(
        "--out-root",
        type=Path,
        default=EVAL_RESULTS_DIR,
        help="Output root directory.",
    )
    
    # Fusion config
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-inner-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Build config
    config = FusionConfig(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_inner_folds=args.n_inner_folds,
        random_seed=args.random_seed,
    )
    
    # Determine pairs to run
    if "all" in args.pairs:
        pairs_to_run = list(FUSION_PAIRS.keys())
    else:
        pairs_to_run = args.pairs
    
    print("=" * 70)
    print("TAL Fusion: V-JEPA + Skeleton Models")
    print("=" * 70)
    print(f"Pairs to run: {pairs_to_run}")
    print(f"Output root: {args.out_root}")
    print(f"Config: hidden_dim={config.hidden_dim}, epochs={config.num_epochs}, lr={config.lr}")
    print()
    
    # Run fusion for each pair
    all_summaries = {}
    
    for pair_key in pairs_to_run:
        summary = run_fusion_for_pair(
            pair_key=pair_key,
            folds=args.folds,
            config=config,
            out_root=args.out_root,
        )
        all_summaries[pair_key] = summary
    
    # Save overall summary
    summary_path = args.out_root / "fusion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for pair_key, summary in all_summaries.items():
        print(f"\n{summary['description']}:")
        print(f"  Output: {summary['output_dir']}")
        print(f"  Folds: {summary['folds_processed']}")
        if "mean_oof_accuracy" in summary:
            print(f"  Mean OOF accuracy: {summary['mean_oof_accuracy']:.4f}")
    
    print(f"\nFusion summary saved to: {summary_path}")
    print("\nDone! Run grid search on fused predictions with:")
    print("  python scripts/grid_search_postprocessing.py")


if __name__ == "__main__":
    main()



