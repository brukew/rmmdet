#!/usr/bin/env python3
"""
Parse SAM3 labels and notes from rmm_sam3.csv into time-based ID mappings.

Rules:
- Multiple IDs with no notes or notes like "unique ids", "both the whole time" → all IDs for whole video
- Time-based notes describe when specific IDs are active
- When an ID is described as "until X then again at Y", the OTHER IDs are active from X to Y
- Empty SAM3 labels are skipped
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Resolve paths relative to the repo root so this runs from any clone location.
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "paths.py").exists())


def parse_timestamp(ts: str) -> Optional[float]:
    """
    Parse timestamp like "0.11", "0:11", "1.07", "1:07" into seconds.
    
    Based on context, these appear to be MM:SS format written as M.SS
    (e.g., 0.11 = 0:11 = 11 seconds, 1.07 = 1:07 = 67 seconds)
    """
    if ts is None:
        return None
    
    ts = ts.strip()
    if not ts:
        return None
    
    # Handle "X" or other non-numeric values
    if not re.match(r'^[\d:.]+$', ts):
        return None
    
    # If contains colon, parse as MM:SS
    if ':' in ts:
        parts = ts.split(':')
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    
    # If contains dot, interpret as M.SS format (e.g., 0.11 = 0:11 = 11 seconds)
    if '.' in ts:
        try:
            parts = ts.split('.')
            if len(parts) == 2:
                minutes = int(parts[0])
                sec_str = parts[1]
                if len(sec_str) <= 2:
                    seconds = int(sec_str)
                else:
                    seconds = float('0.' + sec_str) * 100
                return minutes * 60 + seconds
        except ValueError:
            pass
    
    # Plain number - assume seconds
    try:
        return float(ts)
    except ValueError:
        return None


def parse_sam3_label(label: str) -> List[str]:
    """
    Parse SAM3 label into list of IDs.
    
    Handles:
    - "0" -> ["0"]
    - "1,3" -> ["1", "3"]
    - "0.2" -> ["0", "2"] (dotted notation for ID transition)
    - "1, 2, X" -> ["1", "2", "X"]
    """
    if not label or label.strip() == '':
        return []
    
    label = label.strip().strip('"')
    
    # Check for dotted notation like "0.2" meaning start=0, becomes=2
    if re.match(r'^\d+\.\d+$', label) and ',' not in label:
        parts = label.split('.')
        return parts
    
    # Split by comma
    ids = []
    for part in re.split(r'[,\s]+', label):
        part = part.strip()
        if part:
            ids.append(part)
    
    return ids


def is_whole_video_note(notes: str) -> bool:
    """Check if notes indicate all IDs should be used for whole video."""
    if not notes:
        return True
    
    notes_lower = notes.lower()
    whole_video_patterns = [
        'both the whole time',
        'both at same time',
        'all at once',
        'unique ids',
        'check again',
        'mask a bit spotty',
        'rough mask',
        'watch out',
    ]
    
    for pattern in whole_video_patterns:
        if pattern in notes_lower:
            return True
    
    # If no time-related patterns found, assume whole video
    time_patterns = [
        r'at\s+\d',
        r'until\s+\d',
        r'become',
        r'then\s+\d',
    ]
    
    has_time_info = any(re.search(p, notes_lower) for p in time_patterns)
    return not has_time_info


def parse_time_transitions(notes: str, ids: List[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse notes to extract time-based ID transitions.
    
    Returns list of intervals: {ids: [list of active IDs], start_sec: float, end_sec: float|None}
    """
    if not notes or is_whole_video_note(notes):
        return None
    
    notes_lower = notes.lower()
    intervals = []
    
    # Pattern: "X until T then X again at T2" 
    # This means: ID X from 0 to T, other IDs from T to T2, ID X from T2 onwards
    until_again_pattern = r'(\d+)\s+until\s+([\d.:]+)\s+then\s+\1\s+again\s+at\s+([\d.:]+)'
    match = re.search(until_again_pattern, notes_lower)
    if match:
        main_id = match.group(1)
        until_time = parse_timestamp(match.group(2))
        again_time = parse_timestamp(match.group(3))
        
        if until_time is not None and again_time is not None:
            other_ids = [i for i in ids if i != main_id and i.upper() != 'X']
            
            intervals.append({
                'ids': [main_id],
                'start_sec': 0.0,
                'end_sec': until_time,
            })
            if other_ids:
                intervals.append({
                    'ids': other_ids,
                    'start_sec': until_time,
                    'end_sec': again_time,
                })
            intervals.append({
                'ids': [main_id],
                'start_sec': again_time,
                'end_sec': None,
            })
            return intervals
    
    # Pattern: "becomes X at T" or "become X at T"
    become_pattern = r'becomes?\s+(\d+)\s+at\s+([\d.:]+)'
    become_matches = list(re.finditer(become_pattern, notes_lower))
    
    if become_matches:
        # First ID until the first transition
        first_time = parse_timestamp(become_matches[0].group(2))
        if first_time is not None:
            # Find the first ID (the one that becomes something else)
            first_id = ids[0] if ids else None
            if first_id:
                intervals.append({
                    'ids': [first_id],
                    'start_sec': 0.0,
                    'end_sec': first_time,
                })
            
            # Each "becomes X" creates a new interval
            for i, match in enumerate(become_matches):
                new_id = match.group(1)
                start_time = parse_timestamp(match.group(2))
                
                # End time is either the next transition or None (end of video)
                if i + 1 < len(become_matches):
                    end_time = parse_timestamp(become_matches[i + 1].group(2))
                else:
                    end_time = None
                
                if start_time is not None:
                    intervals.append({
                        'ids': [new_id],
                        'start_sec': start_time,
                        'end_sec': end_time,
                    })
            
            return intervals if intervals else None
    
    # Pattern: "X at T, then Y at T2, then Z at T3"
    # More complex transitions
    at_pattern = r'(\d+|X)\s+at\s+([\d.:]+)'
    at_matches = list(re.finditer(at_pattern, notes_lower))
    
    if at_matches:
        # Start with first ID until first transition
        first_time = parse_timestamp(at_matches[0].group(2))
        if first_time is not None and ids:
            # The first ID is active until the first mentioned time
            first_id = ids[0]
            intervals.append({
                'ids': [first_id],
                'start_sec': 0.0,
                'end_sec': first_time,
            })
        
        for i, match in enumerate(at_matches):
            id_val = match.group(1)
            start_time = parse_timestamp(match.group(2))
            
            # End time is the next transition or None
            if i + 1 < len(at_matches):
                end_time = parse_timestamp(at_matches[i + 1].group(2))
            else:
                end_time = None
            
            if start_time is not None and id_val.upper() != 'X':
                intervals.append({
                    'ids': [id_val],
                    'start_sec': start_time,
                    'end_sec': end_time,
                })
        
        return intervals if intervals else None
    
    return None


def build_final_intervals(ids: List[str], transitions: Optional[List[Dict]]) -> List[Dict[str, Any]]:
    """
    Build final interval list.
    
    If no transitions, all IDs apply for whole video.
    Otherwise, use the parsed transitions.
    """
    # Filter out X (invalid/missing) IDs
    valid_ids = [i for i in ids if i.upper() != 'X']
    
    if not valid_ids:
        return []
    
    # No transitions - all IDs for whole video
    if transitions is None:
        return [{
            'id': i,
            'start_sec': 0.0,
            'end_sec': None,
        } for i in valid_ids]
    
    # Flatten transitions into per-ID intervals
    intervals = []
    for trans in transitions:
        for id_val in trans['ids']:
            if id_val.upper() != 'X':
                intervals.append({
                    'id': id_val,
                    'start_sec': trans['start_sec'],
                    'end_sec': trans['end_sec'],
                })
    
    return intervals


def process_csv(input_path: Path, output_path: Path) -> None:
    """Process the SAM3 CSV and output parsed time intervals."""
    rows_out = []
    issues = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        # Skip the "Table 1" header line if present
        first_line = f.readline()
        if not first_line.startswith('SAM3'):
            pass  # It was a header line
        else:
            f.seek(0)
        
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        for row_idx, row in enumerate(reader, start=1):
            sam3_label = row.get('SAM3 label', '').strip()
            sam3_notes = row.get('SAM3 Notes', '').strip()
            filename = row.get('FileName', '')
            
            # Parse label into IDs
            ids = parse_sam3_label(sam3_label)
            
            # Skip rows with no IDs
            if not ids:
                row['child_sam3_ids'] = '[]'
                row['parsed_ids'] = ''
                row['has_transitions'] = 'no'
                row['parse_status'] = 'skipped_no_ids'
                rows_out.append(row)
                continue
            
            # Parse notes for transitions
            transitions = parse_time_transitions(sam3_notes, ids)
            
            # Build final intervals
            intervals = build_final_intervals(ids, transitions)
            
            # Determine parse status
            if transitions is not None:
                parse_status = 'has_transitions'
            elif len(ids) > 1:
                parse_status = 'multiple_ids_whole_video'
            else:
                parse_status = 'single_id'
            
            # Add parsed data to row
            row['child_sam3_ids'] = json.dumps(intervals)
            row['parsed_ids'] = ','.join(ids)
            row['has_transitions'] = 'yes' if transitions else 'no'
            row['parse_status'] = parse_status
            
            rows_out.append(row)
    
    # Write output
    out_fieldnames = list(fieldnames) + ['parsed_ids', 'has_transitions', 'parse_status', 'child_sam3_ids']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({k: row.get(k, '') for k in out_fieldnames})
    
    # Print summary
    total = len(rows_out)
    with_ids = sum(1 for r in rows_out if r.get('parsed_ids'))
    with_transitions = sum(1 for r in rows_out if r.get('has_transitions') == 'yes')
    skipped = sum(1 for r in rows_out if r.get('parse_status') == 'skipped_no_ids')
    
    print(f"Processed {total} rows")
    print(f"  - With SAM3 IDs: {with_ids}")
    print(f"  - With time transitions: {with_transitions}")
    print(f"  - Skipped (no IDs): {skipped}")
    print(f"Output: {output_path}")
    
    # Print rows with transitions for verification
    print(f"\n{'='*60}")
    print("ROWS WITH TIME TRANSITIONS (verify these):")
    print(f"{'='*60}")
    for row in rows_out:
        if row.get('has_transitions') == 'yes':
            print(f"\nRow: {row.get('FileName', 'unknown')}")
            print(f"  SAM3 label: {row.get('SAM3 label', '')}")
            print(f"  SAM3 Notes: {row.get('SAM3 Notes', '')}")
            intervals = json.loads(row.get('child_sam3_ids', '[]'))
            print(f"  Intervals:")
            for iv in intervals:
                end = iv.get('end_sec') or 'end'
                print(f"    ID {iv['id']}: {iv['start_sec']:.1f}s - {end}")


def main():
    parser = argparse.ArgumentParser(description="Parse SAM3 labels into time-based ID intervals")
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('/orcd/data/satra/001/users/brukew/sailsprep/subset_data/rmm_sam3.csv'),
        help='Input CSV with SAM3 labels',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=_REPO_ROOT / "dataprep/rmm_sam3_parsed.csv",
        help='Output CSV with parsed time intervals',
    )
    args = parser.parse_args()
    
    process_csv(args.input, args.output)


if __name__ == '__main__':
    main()
