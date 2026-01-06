#!/usr/bin/env python3
"""
Convert prediction CSVs from pyskl or V-JEPA format to standardized TAL eval format.

Supports:
- pyskl predictions_clip.csv (segment_id with embedded video_key and timestamps)
- V-JEPA window_level_preds.csv (window_id with embedded timestamps, separate video_key)

Output format (TAL eval standard):
    window_id, video_key, start_sec, end_sec, score_class0, score_class1, ...

The video_key must match the format used by GT segments (full path like 
'A.B._Home_Videos_AMES_L5V8G4A3K7/12-16 month videos/4-10-2020.mov').
For pyskl, this is obtained via --window-csv lookup.
For V-JEPA, it's already in the correct format.

Example usage:
    # Convert pyskl predictions (requires window CSV for video_key mapping)
    python convert_preds_to_tal_format.py \
        -i predictions_clip.csv \
        -o tal_format_preds.csv \
        --window-csv fold_0_val_windows.csv

    # Convert V-JEPA predictions (video_key already correct)
    python convert_preds_to_tal_format.py \
        -i window_level_preds.csv \
        -o tal_format_preds.csv
        
    # Fuse multiple modality CSVs (for STGCN++ 4-stream)
    python convert_preds_to_tal_format.py \
        -i j_preds.csv b_preds.csv jm_preds.csv bm_preds.csv \
        -o fused_preds.csv \
        --fuse \
        --window-csv fold_0_val_windows.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_timestamps(id_str: str) -> Tuple[str, float, float]:
    """
    Parse video_key_compact, start_sec, end_sec from segment_id or window_id.
    
    Format: {video_key_compact}__t{start_ms}_{end_ms}
    Example: 'L5V8G4A3K7_4-10-2020__t0_2000' -> ('L5V8G4A3K7_4-10-2020', 0.0, 2.0)
    
    Args:
        id_str: segment_id or window_id string
        
    Returns:
        Tuple of (video_key_compact, start_sec, end_sec)
    """
    # Split on '__t' to separate video_key from timestamp part
    parts = id_str.rsplit('__t', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid id format (missing '__t'): {id_str}")
    
    video_key_compact = parts[0]
    time_part = parts[1]  # e.g., "0_2000"
    
    # Split time part on '_' to get start and end milliseconds
    time_parts = time_part.split('_')
    if len(time_parts) != 2:
        raise ValueError(f"Invalid time format (expected 'start_end'): {time_part}")
    
    start_ms = float(time_parts[0])
    end_ms = float(time_parts[1])
    
    return video_key_compact, start_ms / 1000.0, end_ms / 1000.0


def build_window_id_to_video_key_map(window_csv: Path) -> Dict[str, str]:
    """
    Build a mapping from window_id to full video_key (path format).
    
    Args:
        window_csv: Path to window CSV with window_id and video_key columns.
        
    Returns:
        Dict mapping window_id -> video_key (full path format)
    """
    df = pd.read_csv(window_csv)
    if 'window_id' not in df.columns or 'video_key' not in df.columns:
        raise ValueError(f"Window CSV must have 'window_id' and 'video_key' columns: {window_csv}")
    
    return dict(zip(df['window_id'], df['video_key']))


def detect_format(df: pd.DataFrame) -> str:
    """
    Detect whether DataFrame is from pyskl or V-JEPA based on columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        'pyskl' or 'vjepa'
    """
    if 'segment_id' in df.columns:
        return 'pyskl'
    elif 'window_id' in df.columns:
        return 'vjepa'
    else:
        raise ValueError("Cannot detect format: missing 'segment_id' or 'window_id' column")


def convert_to_tal_format(
    df: pd.DataFrame,
    window_id_to_video_key: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Convert prediction DataFrame to standardized TAL eval format.
    
    Args:
        df: Input DataFrame (pyskl or V-JEPA format)
        window_id_to_video_key: Optional mapping from window_id to full video_key.
                                Required for pyskl format to get correct video paths.
        
    Returns:
        DataFrame with columns: window_id, video_key, start_sec, end_sec, score_class*
    """
    fmt = detect_format(df)
    
    # Get the ID column name
    id_col = 'segment_id' if fmt == 'pyskl' else 'window_id'
    
    # Parse timestamps from ID column
    parsed = df[id_col].apply(parse_timestamps)
    
    # Create output DataFrame
    out_df = pd.DataFrame()
    out_df['window_id'] = df[id_col]
    out_df['start_sec'] = [p[1] for p in parsed]
    out_df['end_sec'] = [p[2] for p in parsed]
    
    # Get video_key - either from mapping or from V-JEPA's existing column
    if fmt == 'pyskl':
        if window_id_to_video_key is None:
            # Fallback to compact format (will likely fail GT matching)
            print("  WARNING: No window CSV provided. Using compact video_key (may not match GT).")
            out_df['video_key'] = [p[0] for p in parsed]
        else:
            # Use mapping to get full video_key path
            out_df['video_key'] = out_df['window_id'].map(window_id_to_video_key)
            missing = out_df['video_key'].isna().sum()
            if missing > 0:
                print(f"  WARNING: {missing} window_ids not found in mapping")
    else:
        # V-JEPA already has video_key in correct format
        out_df['video_key'] = df['video_key']
    
    # Copy score columns
    score_cols = [c for c in df.columns if c.startswith('score_class')]
    for col in sorted(score_cols):
        out_df[col] = df[col]
    
    return out_df


def fuse_predictions(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Fuse multiple prediction DataFrames by averaging scores.
    
    All DataFrames must have the same window_ids in the same order.
    
    Args:
        dfs: List of DataFrames in TAL format (window_id, video_key, start_sec, end_sec, score_class*)
        
    Returns:
        Fused DataFrame with averaged scores
    """
    if len(dfs) == 1:
        return dfs[0]
    
    # Use first DataFrame as base
    base_df = dfs[0].copy()
    
    # Verify all DataFrames have the same window_ids
    for i, df in enumerate(dfs[1:], start=2):
        if not (df['window_id'] == base_df['window_id']).all():
            raise ValueError(f"DataFrame {i} has different window_ids than DataFrame 1")
    
    # Get score columns
    score_cols = [c for c in base_df.columns if c.startswith('score_class')]
    
    # Average scores across all DataFrames
    for col in score_cols:
        base_df[col] = sum(df[col] for df in dfs) / len(dfs)
    
    return base_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert prediction CSVs to standardized TAL eval format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        nargs='+',
        type=Path,
        required=True,
        help="Input CSV file(s). Multiple files for fusion.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output CSV file in TAL format.",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse multiple input CSVs by averaging scores (for multi-stream fusion).",
    )
    parser.add_argument(
        "--window-csv",
        type=Path,
        default=None,
        help="Window CSV with window_id -> video_key mapping. Required for pyskl format.",
    )
    
    args = parser.parse_args()
    
    # Build window_id -> video_key mapping if provided
    window_id_to_video_key = None
    if args.window_csv:
        print(f"Loading window mapping from: {args.window_csv}")
        window_id_to_video_key = build_window_id_to_video_key_map(args.window_csv)
        print(f"  Loaded {len(window_id_to_video_key)} window_id -> video_key mappings")
    
    # Load and convert input CSVs
    converted_dfs = []
    for input_path in args.input:
        print(f"Loading: {input_path}")
        df = pd.read_csv(input_path)
        fmt = detect_format(df)
        print(f"  Detected format: {fmt}")
        print(f"  Rows: {len(df)}")
        
        converted = convert_to_tal_format(df, window_id_to_video_key)
        converted_dfs.append(converted)
        print(f"  Converted to TAL format")
    
    # Fuse if multiple inputs
    if args.fuse and len(converted_dfs) > 1:
        print(f"\nFusing {len(converted_dfs)} streams by averaging scores...")
        output_df = fuse_predictions(converted_dfs)
    elif len(converted_dfs) > 1 and not args.fuse:
        raise ValueError("Multiple input files provided but --fuse not specified")
    else:
        output_df = converted_dfs[0]
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")
    print(f"  Rows: {len(output_df)}")
    
    # Show sample
    score_cols = [c for c in output_df.columns if c.startswith('score_class')]
    print(f"  Score columns: {score_cols}")
    print(f"\nSample output:")
    print(output_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()

