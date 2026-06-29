#!/usr/bin/env python3
"""
Generate JSON files for all paper tables by extracting data from markdown files.

Tables generated:
- Table 5.1: 4-class Classification Best per Family
- Table 5.2: 5-class Classification Best per Family
- Table 5.5: Window-level Detection Best per Family
- Table 5.7: Segment-level TAL mAP
- Table 5.9: Binary TAL Performance
"""

import json
import re
from pathlib import Path

ACTREG_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = ACTREG_ROOT / "insights" / "tables"


def parse_percent(text):
    """Parse percentage value, handling ± notation."""
    text = text.replace("**", "").strip()
    match = re.search(r"([\d.]+)%\s*(?:±\s*([\d.]+)%)?", text)
    if match:
        mean = float(match.group(1))
        std = float(match.group(2)) if match.group(2) else None
        return {"mean": mean, "std": std}
    return None


def parse_kappa(text):
    """Parse Cohen's kappa value."""
    text = text.replace("**", "").strip()
    match = re.search(r"([\d.]+)\s*(?:±\s*([\d.]+))?", text)
    if match:
        mean = float(match.group(1))
        std = float(match.group(2)) if match.group(2) else None
        return {"mean": mean, "std": std}
    return None


def extract_table_5_1():
    """Extract 4-class Classification Best per Family."""
    md_file = ACTREG_ROOT / "MODEL_COMPARISON.md"
    with open(md_file) as f:
        content = f.read()
    
    # Find the 4-Class Task table
    pattern = r"### 4-Class Task \(Clip-Level Metrics\)\s*\n\n(.*?)\n\n\*Note"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find 4-class table")
        return {}
    
    table_text = match.group(1)
    results = {"task": "4-class_clip_level", "models": []}
    
    for line in table_text.strip().split("\n"):
        if "|" not in line or "---" in line or "Rank" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        
        if len(parts) < 8:
            continue
        
        rank = parts[0].replace("🥇", "1").replace("🥈", "2").replace("🥉", "3").strip()
        method = parts[1].replace("**", "").strip()
        
        model_data = {
            "rank": int(rank) if rank.isdigit() else rank,
            "method": method,
            "clip_top1": parse_percent(parts[2]),
            "clip_top2": parse_percent(parts[3]),
            "macro_f1": parse_percent(parts[4]),
            "macro_precision": parse_percent(parts[5]),
            "macro_recall": parse_percent(parts[6]),
            "cohens_kappa": parse_kappa(parts[7])
        }
        results["models"].append(model_data)
    
    return results


def extract_table_5_2():
    """Extract 5-class Classification Best per Family."""
    md_file = ACTREG_ROOT / "MODEL_COMPARISON.md"
    with open(md_file) as f:
        content = f.read()
    
    # Find the 5-Class Task table
    pattern = r"### 5-Class Task \(Clip-Level Metrics\)\s*\n\n(.*?)\n\n\*Note"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find 5-class table")
        return {}
    
    table_text = match.group(1)
    results = {"task": "5-class_clip_level", "models": []}
    
    for line in table_text.strip().split("\n"):
        if "|" not in line or "---" in line or "Rank" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        
        if len(parts) < 8:
            continue
        
        rank = parts[0].replace("🥇", "1").replace("🥈", "2").replace("🥉", "3").strip()
        method = parts[1].replace("**", "").strip()
        
        model_data = {
            "rank": int(rank) if rank.isdigit() else rank,
            "method": method,
            "clip_top1": parse_percent(parts[2]),
            "clip_top2": parse_percent(parts[3]),
            "macro_f1": parse_percent(parts[4]),
            "macro_precision": parse_percent(parts[5]),
            "macro_recall": parse_percent(parts[6]),
            "cohens_kappa": parse_kappa(parts[7])
        }
        results["models"].append(model_data)
    
    return results


def extract_table_5_5():
    """Extract Window-level Detection Best per Family."""
    md_file = ACTREG_ROOT / "tal" / "TAL_DET_MODEL_COMPARISON.md"
    with open(md_file) as f:
        content = f.read()
    
    # Find the Summary Table
    pattern = r"### Summary Table\s*\n\n(.*?)\n\n\*"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print("Warning: Could not find window-level detection table")
        return {}
    
    table_text = match.group(1)
    results = {"task": "window_level_detection_5class", "models": []}
    
    for line in table_text.strip().split("\n"):
        if "|" not in line or "---" in line or "Model" in line:
            continue
        
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        
        if len(parts) < 6:
            continue
        
        model = parts[0].replace("**", "").replace("🏆", "").strip()
        
        model_data = {
            "model": model,
            "folds": parts[1].strip(),
            "top1_acc": parse_percent(parts[2]),
            "top2_acc": parse_percent(parts[3]),
            "macro_f1": parse_percent(parts[4]),
            "cohens_kappa": parse_kappa(parts[5]) if len(parts) > 5 else None
        }
        results["models"].append(model_data)
    
    return results


def extract_table_5_7():
    """Extract Segment-level TAL mAP."""
    md_file = ACTREG_ROOT / "tal" / "TAL_MODEL_COMPARISON.md"
    with open(md_file) as f:
        content = f.read()
    
    # Find the late fusion results and single model results
    results = {"task": "segment_level_tal", "models": []}
    
    # Late Fusion Results
    pattern_fusion = r"### Fusion Results.*?\n\n(.*?)\n\n###"
    match_fusion = re.search(pattern_fusion, content, re.DOTALL)
    
    if match_fusion:
        for line in match_fusion.group(1).strip().split("\n"):
            if "|" not in line or "---" in line or "Model" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            
            if len(parts) >= 6:
                model_data = {
                    "model": parts[0].replace("**", "").strip(),
                    "folds": parts[1].strip() if len(parts) > 1 else None,
                    "mAP_0.3": parse_percent(parts[2]) if len(parts) > 2 else None,
                    "mAP_0.5": parse_percent(parts[3]) if len(parts) > 3 else None,
                    "mAP_0.7": parse_percent(parts[4]) if len(parts) > 4 else None,
                    "avg_mAP": parse_percent(parts[5]) if len(parts) > 5 else None
                }
                results["models"].append(model_data)
    
    # ActionFormer results
    pattern_af = r"### ActionFormer \+ V-JEPA Balanced.*?\n\n(.*?)\n\n###"
    match_af = re.search(pattern_af, content, re.DOTALL)
    
    if match_af:
        for line in match_af.group(1).strip().split("\n"):
            if "|" not in line or "---" in line or "Fold" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            
            if "Mean" in parts[0] and len(parts) >= 6:
                model_data = {
                    "model": "ActionFormer + V-JEPA Balanced",
                    "mAP_0.3": parse_percent(parts[1]) if len(parts) > 1 else None,
                    "mAP_0.4": parse_percent(parts[2]) if len(parts) > 2 else None,
                    "mAP_0.5": parse_percent(parts[3]) if len(parts) > 3 else None,
                    "mAP_0.6": parse_percent(parts[4]) if len(parts) > 4 else None,
                    "mAP_0.7": parse_percent(parts[5]) if len(parts) > 5 else None,
                    "avg_mAP": parse_percent(parts[6]) if len(parts) > 6 else None
                }
                results["models"].append(model_data)
    
    return results


def extract_table_5_9():
    """Extract Binary TAL Performance."""
    md_file = ACTREG_ROOT / "tal" / "TAL_MODEL_COMPARISON.md"
    with open(md_file) as f:
        content = f.read()
    
    results = {"task": "binary_tal", "models": []}
    
    # ActionFormer Binary results
    pattern = r"### ActionFormer \+ V-JEPA Binary.*?\n\n(.*?)\n\n###"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        for line in match.group(1).strip().split("\n"):
            if "|" not in line or "---" in line or "Fold" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            
            if "Mean" in parts[0] and len(parts) >= 6:
                model_data = {
                    "model": "ActionFormer + V-JEPA Binary",
                    "mAP_0.3": parse_percent(parts[1]) if len(parts) > 1 else None,
                    "mAP_0.4": parse_percent(parts[2]) if len(parts) > 2 else None,
                    "mAP_0.5": parse_percent(parts[3]) if len(parts) > 3 else None,
                    "mAP_0.6": parse_percent(parts[4]) if len(parts) > 4 else None,
                    "mAP_0.7": parse_percent(parts[5]) if len(parts) > 5 else None,
                    "avg_mAP": parse_percent(parts[6]) if len(parts) > 6 else None
                }
                results["models"].append(model_data)
    
    # Also get comparison table
    pattern_comp = r"### ActionFormer vs Window-Based TAL Comparison\s*\n\n(.*?)\n\n"
    match_comp = re.search(pattern_comp, content, re.DOTALL)
    
    if match_comp:
        for line in match_comp.group(1).strip().split("\n"):
            if "|" not in line or "---" in line or "Model" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]
            
            if len(parts) >= 6 and "Binary" in parts[0]:
                model_data = {
                    "model": parts[0].replace("**", "").strip(),
                    "type": parts[1].strip(),
                    "mAP_0.3": parse_percent(parts[2]),
                    "mAP_0.5": parse_percent(parts[3]),
                    "mAP_0.7": parse_percent(parts[4]),
                    "avg_mAP": parse_percent(parts[5])
                }
                results["models"].append(model_data)
    
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all table JSONs
    tables = {
        "table_5_1_4class_classification.json": extract_table_5_1(),
        "table_5_2_5class_classification.json": extract_table_5_2(),
        "table_5_5_window_detection.json": extract_table_5_5(),
        "table_5_7_segment_tal_map.json": extract_table_5_7(),
        "table_5_9_binary_tal.json": extract_table_5_9()
    }
    
    for filename, data in tables.items():
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {output_path}")
        print(f"  Models: {len(data.get('models', []))}")


if __name__ == "__main__":
    main()
