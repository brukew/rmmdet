#!/usr/bin/env python3
"""
Tables 5.3 & 5.4: Per-class Precision/Recall for Classification.

Extracts per-class P/R data from MODEL_COMPARISON.md for both 4-class and 5-class tasks.
These tables show precision/recall for the best model from each family.

Usage:
    python tables_5_3_5_4_classification_pr.py
    python tables_5_3_5_4_classification_pr.py --output table_5_3_4class_pr.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

# Project root
ACTREG_ROOT = Path(__file__).parent.parent.parent

# Paths
MODEL_COMPARISON_FILE = ACTREG_ROOT / "MODEL_COMPARISON.md"
OUTPUT_DIR = ACTREG_ROOT / "insights" / "paper_plots"

# Class names
CLASSES_4CLASS = ["hands flapping", "jumping", "rocking", "spinning"]
CLASSES_5CLASS = ["hands flapping", "jumping", "one hand flap", "rocking", "spinning"]


def parse_pr_table(markdown_content: str, section_header: str) -> Dict:
    """
    Parse a P/R table from markdown content.
    
    Args:
        markdown_content: Full markdown file content
        section_header: Section header to find (e.g., "### 4-Class Task")
        
    Returns:
        Dict mapping model_family -> {class -> {"P": val, "R": val}}
    """
    # Find the section
    pattern = rf"{re.escape(section_header)}.*?\n\n(.*?)\n\n"
    match = re.search(pattern, markdown_content, re.DOTALL)
    
    if not match:
        print(f"Warning: Section '{section_header}' not found")
        return {}
    
    table_text = match.group(1)
    
    # Parse table rows
    results = {}
    lines = table_text.strip().split("\n")
    
    # Skip header and separator lines
    for line in lines:
        if "|" not in line or "---" in line or "Model Family" in line:
            continue
        
        # Parse row
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]  # Remove empty parts
        
        if len(parts) < 4:
            continue
        
        model_family = parts[0].replace("**", "").strip()
        model_name = parts[1].replace("**", "").strip()
        
        # Parse P/R pairs (format: "XX.X% / YY.Y%")
        class_data = {}
        for i, class_name in enumerate(CLASSES_4CLASS if "4-Class" in section_header else CLASSES_5CLASS):
            if i + 2 < len(parts):
                pr_text = parts[i + 2]
                # Handle bold markers
                pr_text = pr_text.replace("**", "")
                
                # Parse P/R
                pr_match = re.search(r"(\d+\.?\d*)%\s*/\s*(\d+\.?\d*)%", pr_text)
                if pr_match:
                    class_data[class_name] = {
                        "precision": float(pr_match.group(1)),
                        "recall": float(pr_match.group(2))
                    }
        
        if class_data:
            results[model_family] = {
                "model_name": model_name,
                "per_class": class_data
            }
    
    return results


def extract_classification_pr() -> Dict:
    """
    Extract per-class P/R data from MODEL_COMPARISON.md.
    
    Returns:
        Dict with "4class" and "5class" keys containing model data
    """
    with open(MODEL_COMPARISON_FILE) as f:
        content = f.read()
    
    # Find the "Per-Class Precision/Recall (Best Models)" section
    # Look for the 4-class and 5-class tables
    
    results = {"4class": {}, "5class": {}}
    
    # Parse 4-class table
    # The table is under "### 4-Class Task" within "## Per-Class Precision/Recall"
    pattern_4class = r"## Per-Class Precision/Recall \(Best Models\).*?### 4-Class Task\s*\n\n(.*?)\n\n\*\*Key"
    match_4class = re.search(pattern_4class, content, re.DOTALL)
    
    if match_4class:
        table_4class = match_4class.group(1)
        results["4class"] = parse_table_rows(table_4class, CLASSES_4CLASS)
    
    # Parse 5-class table
    pattern_5class = r"### 5-Class Task\s*\n\n(.*?)\n\n\*\*Key"
    match_5class = re.search(pattern_5class, content, re.DOTALL)
    
    if match_5class:
        table_5class = match_5class.group(1)
        results["5class"] = parse_table_rows(table_5class, CLASSES_5CLASS)
    
    return results


def parse_table_rows(table_text: str, class_names: List[str]) -> Dict:
    """Parse markdown table rows into structured data."""
    results = {}
    
    for line in table_text.strip().split("\n"):
        if "|" not in line or "---" in line or "Model Family" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        
        if len(parts) < 4:
            continue
        
        model_family = parts[0].replace("**", "").strip()
        model_name = parts[1].replace("**", "").strip()
        
        per_class = {}
        for i, class_name in enumerate(class_names):
            if i + 2 < len(parts):
                pr_text = parts[i + 2].replace("**", "")
                pr_match = re.search(r"(\d+\.?\d*)%\s*/\s*(\d+\.?\d*)%", pr_text)
                if pr_match:
                    per_class[class_name] = {
                        "precision": float(pr_match.group(1)),
                        "recall": float(pr_match.group(2))
                    }
        
        if per_class:
            results[model_family] = {
                "model_name": model_name,
                "per_class": per_class
            }
    
    return results


def print_table_5_3(data: Dict):
    """Print Table 5.3: Per-class P/R for 4-class classification."""
    print("\n" + "="*80)
    print("Table 5.3: Per-class Precision/Recall for 4-Class Classification")
    print("="*80)
    
    print("\n| Model Family | Model | " + " | ".join([f"{c} P/R" for c in CLASSES_4CLASS]) + " |")
    print("|" + "|".join(["------"] * (len(CLASSES_4CLASS) + 2)) + "|")
    
    for family, info in data["4class"].items():
        row = [family, info["model_name"]]
        for c in CLASSES_4CLASS:
            if c in info["per_class"]:
                p = info["per_class"][c]["precision"]
                r = info["per_class"][c]["recall"]
                row.append(f"{p:.1f}% / {r:.1f}%")
            else:
                row.append("N/A")
        print("| " + " | ".join(row) + " |")


def print_table_5_4(data: Dict):
    """Print Table 5.4: Per-class P/R for 5-class classification."""
    print("\n" + "="*80)
    print("Table 5.4: Per-class Precision/Recall for 5-Class Classification")
    print("="*80)
    
    print("\n| Model Family | Model | " + " | ".join([f"{c} P/R" for c in CLASSES_5CLASS]) + " |")
    print("|" + "|".join(["------"] * (len(CLASSES_5CLASS) + 2)) + "|")
    
    for family, info in data["5class"].items():
        row = [family, info["model_name"]]
        for c in CLASSES_5CLASS:
            if c in info["per_class"]:
                p = info["per_class"][c]["precision"]
                r = info["per_class"][c]["recall"]
                row.append(f"{p:.1f}% / {r:.1f}%")
            else:
                row.append("N/A")
        print("| " + " | ".join(row) + " |")


def save_results(data: Dict, output_dir: Path):
    """Save extracted data to JSON files."""
    # Save 4-class data
    json_4class = output_dir / "table_5_3_4class_pr.json"
    with open(json_4class, 'w') as f:
        json.dump(data["4class"], f, indent=2)
    print(f"\nSaved: {json_4class}")
    
    # Save 5-class data
    json_5class = output_dir / "table_5_4_5class_pr.json"
    with open(json_5class, 'w') as f:
        json.dump(data["5class"], f, indent=2)
    print(f"Saved: {json_5class}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-class P/R for classification (Tables 5.3 & 5.4)"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Extract data
    print("Extracting per-class P/R from MODEL_COMPARISON.md...")
    data = extract_classification_pr()
    
    if not data["4class"] and not data["5class"]:
        print("Warning: No data extracted. Check if MODEL_COMPARISON.md format has changed.")
        return
    
    print(f"Found {len(data['4class'])} models for 4-class task")
    print(f"Found {len(data['5class'])} models for 5-class task")
    
    # Print tables
    if data["4class"]:
        print_table_5_3(data)
    
    if data["5class"]:
        print_table_5_4(data)
    
    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    save_results(data, output_dir)


if __name__ == "__main__":
    main()
