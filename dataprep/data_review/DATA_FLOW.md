# Data Flow Analysis: RMM Action Recognition Pipeline

This document tracks data flow through each processing stage, documenting where data is dropped and why.

## Processing Stages Overview

1. **Splits Generation** (Stage 1) - ✅ Documented
   - Raw annotations → Segments extraction → Train/Val/Test splits
   
2. **Clip Creation** (Stage 2) - ✅ Documented
   - CSV splits → Video clips extracted from full-length videos
   
2b. **SAM3 Mask Generation** (Stage 2b) - ✅ Documented
   - Full-length videos → SAM3 mask caches (HDF5)
   
2c. **Pre-Processing** (Stage 2c) - ⏳ To be documented
   - SAM3 clips → HRNet keypoint data
   - Keypoint data → PKL files for skeleton models
   
3. **Data Validation via Model Training** (Stage 3) - ⏳ To be documented
   - SAM3 clips → V-JEPA2 usable data
   - PKL data → Skeleton model usable data

---

## Stage 1: Splits Generation

**Source Notebook:** `dataprep/clip_gen/rmm_annot.ipynb`  
**Output Location:** `dataprep/splits/`

### Overview

The splits generation process extracts RMM segments from raw Excel annotations (`RMM_with_ELE_Highlighted.xlsx`) and creates train/val/test splits respecting LCTO (Leave-Cohort-Timepoint-Out) grouping.

### Process Flow

```
Excel Annotations (RMM_with_ELE_Highlighted.xlsx)
    ↓
Parse timestamps and RMM types
    ↓
Standardize RMM labels (map variants to canonical names)
    ↓
Filter to main RMM classes
    ↓
Extract segments (one per temporal annotation)
    ↓
Generate splits (CV and Single)
    ↓
Output CSV files in dataprep/splits/
```

### Key Processing Steps

1. **RMM Type Standardization:**
   - Maps variant names to canonical classes:
     - `arm flapping`, `arms flapping` → `hands flapping`
     - `one arm flap`, `one arm flapping` → `one hand flap`
     - `bouncing`, `bed jumping`, `knee jumping` → `jumping`
     - `twisting` → `rocking`
     - `clapping` → `hands flapping`

2. **Filtering:**
   - Excludes segments marked as `Unsure` or `Debatable` (from Excel cell colors)
   - Filters to main RMM classes: `hands flapping`, `one hand flap`, `jumping`, `rocking`, `spinning`, `clapping`

3. **LCTO Grouping:**
   - Groups segments by `(child_id, timepoint)` pairs
   - Ensures no data leakage: all segments from same LCTO group stay in same split

### Output Files

#### Cross-Validation Splits (3-fold)

**5-Class (`cv_splits/`):**
- `fold_0_train.csv`, `fold_0_val.csv`
- `fold_1_train.csv`, `fold_1_val.csv`
- `fold_2_train.csv`, `fold_2_val.csv`
- `split_metadata.json`

**4-Class (`cv_splits_4class/`):**
- Same structure as 5-class
- Merges `one hand flap` segments into `hands flapping` (does not filter them out)
- Total segment count remains 654

#### Single Splits

**5-Class (`single_split/`):**
- `train.csv`, `val.csv`, `test.csv`
- `split_info.json`

**4-Class (`single_split_4class/`):**
- Same structure as 5-class
- Merges `one hand flap` segments into `hands flapping` (does not filter them out)
- Total segment count remains 654

### Data Statistics

Total Unique Videos with RMM segment(s): 298

#### 5-Class Splits

| Metric | Value |
|--------|-------|
| Total segments | 654 |
| Total LCTO groups | 151 |
| CV folds | 3 |
| CV train per fold | 436 segments (~101 LCTO groups) |
| CV val per fold | 218 segments (~50 LCTO groups) |
| Single train | 489 segments (105 LCTO groups) |
| Single val | 54 segments (18 LCTO groups) |
| Single test | 111 segments (28 LCTO groups) |

#### 4-Class Splits

| Metric | Value |
|--------|-------|
| Total segments | 654 (same as 5-class, but excludes "one hand flap") |
| Total LCTO groups | 151 |
| Classes | `hands flapping`, `jumping`, `rocking`, `spinning` |
| CV train per fold | 436 segments (~101 LCTO groups) |
| CV val per fold | 218 segments (~50 LCTO groups) |

### Class Distribution (5-Class, CV Fold 0)

| Class | Train Segments | Val Segments | Total |
|-------|---------------|--------------|-------|
| hands flapping | ~141 | ~71 | ~212 |
| jumping | ~124 | ~48 | ~172 |
| one hand flap | ~68 | ~26 | ~94 |
| rocking | ~65 | ~15 | ~80 |
| spinning | ~28 | ~15 | ~43 |
| **Total** | **436** | **218** | **654** |

### Data Flow Notes

1. **Segment Extraction:**
   - Source notebook (`rmm_annot.ipynb`) extracts 654 segments from Excel annotations
   - All segments are included in the final splits (no data loss at this stage)
   - Segments are extracted from `RMM_with_ELE_Highlighted.xlsx` with proper filtering for quality labels

2. **4-Class vs 5-Class:**
   - 4-class version merges `one hand flap` segments into `hands flapping`
   - Total segment count remains 654 (segments are merged, not filtered)
   - LCTO groups remain the same (151 groups)
   - Process: `one hand flap` → `hands flapping` (label change only)

3. **Split Consistency:**
   - CV splits maintain consistent segment counts across folds
   - LCTO groups are evenly distributed (~101 train, ~50 val per fold)
   - No overlap between train/val LCTO groups (verified in metadata)

### CSV File Structure

Each CSV file contains the following columns:

- `segment_id`: Unique identifier (format: `{child_id}_{timepoint}_{video_idx}_{rmm_col_idx}_{seg_idx}`)
- `segment_global_id`: Global numeric ID
- `video_id`: Video identifier
- `lcto_group`: LCTO group identifier (`{child_id}_{timepoint}`)
- `child_id`: Child identifier
- `timepoint`: Timepoint (e.g., `14_month`, `36_month`, `unknown`)
- `video_file`: Source video file path
- `filename`: Video filename
- `rmm_type`: Standardized RMM class
- `rmm_type_raw`: Original RMM annotation
- `start_sec`: Segment start time (seconds)
- `end_sec`: Segment end time (seconds)
- `duration`: Segment duration (seconds)
- `quality_rating`: Quality rating (1-5, or None)
- `annotator_label`: Annotator label (e.g., `False_in_Context`, or None)
- `annotator`: Annotator identifier
- `video_duration`: Total video duration (seconds)
- `n_children`: Number of children in video
- `n_adults`: Number of adults in video

---

## Stage 2: Clip Creation

**Source Script:** `dataprep/clip_gen/create_clip_segments.py`  
**Output Location:** `/orcd/scratch/bcs/001/sensein/sails/rmm/classification_clips/`

### Overview

The clip creation process extracts video segments from full-length standardized videos based on timestamps specified in the split CSV files. Each segment is cut using ffmpeg and validated for proper encoding.

### Process Flow

```
CSV Split Files (dataprep/splits/cv_splits/)
    ↓
Read segment_id, start_sec, end_sec, video_file
    ↓
Resolve video path (standardized .mp4 files)
    ↓
Extract clip using ffmpeg (start_sec to end_sec)
    ↓
Validate clip (ffprobe + decord if available)
    ↓
Store in canonical_clips/ (deduplicated)
    ↓
Link to fold-specific subdirectories
    ↓
Generate clip_manifest.csv
```

### Key Processing Steps

1. **CSV Reading:**
   - Reads CSV files from splits directory (default: `dataprep/splits/cv_splits/`)
   - Processes all `fold_*_*.csv` files found in the directory
   - Normalizes field names to handle variations

2. **Video Path Resolution:**
   - Converts CSV video paths (often with .mov/.MOV) to standardized .mp4 paths
   - Looks in `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized`
   - Handles path prefix removal and extension normalization

3. **Clip Extraction:**
   - Uses ffmpeg to cut clips between `start_sec` and `end_sec`
   - Default codec: h264 (re-encodes for precise cuts)
   - Optional: `--include-end-second` includes full final second (default: enabled)
   - Handles zero-length segments by extending to minimum duration (default: 1.0s)

4. **Deduplication:**
   - Creates canonical clips in `canonical_clips/` directory
   - Links canonical clips to fold-specific subdirectories (symlinks or copies)
   - Prevents re-encoding same segment_id across multiple CSV files

5. **Validation:**
   - Checks file existence and non-zero size
   - Validates using ffprobe (checks video stream readability)
   - Optionally validates using decord (if available) to ensure DataLoader compatibility
   - Skips or rebuilds corrupted clips

### Output Structure

```
classification_clips/
├── canonical_clips/          # Deduplicated clips (one per segment_id)
│   ├── {segment_id}.mp4
│   └── ...
├── fold_0_train/             # Symlinks/copies to canonical clips
│   ├── {segment_id}.mp4
│   └── ...
├── fold_0_val/
├── fold_1_train/
├── fold_1_val/
├── fold_2_train/
├── fold_2_val/
└── clip_manifest.csv         # Manifest of all created clips
```

### Manifest CSV Structure

The `clip_manifest.csv` contains:

- `csv_file`: Source CSV file name
- `segment_id`: Unique segment identifier
- `clip_path`: Path to the created clip
- `source_video`: Path to source video file
- `start_sec`, `end_sec`, `duration`: Clip timing information
- `child_id`, `timepoint`: Metadata
- `label`: RMM type label
- `crop_applied`, `crop_box`, etc.: Cropping metadata (if cropping enabled)

### Data Statistics

| Metric | Value |
|--------|-------|
| Expected segments (from splits) | 654 |
| Canonical clips created | 654 |
| Manifest entries | ~1959 (multiple entries per segment across folds) |
| Unique segments in manifest | 653-654 |

### Data Flow Notes

1. **Clip Creation:**
   - All 654 segments from splits should result in 654 canonical clips
   - Each segment appears in multiple fold CSV files, but only one canonical clip is created
   - Manifest tracks all clip references across all CSV files

2. **Validation:**
   - Clips are validated using ffprobe (checks video stream)
   - Optionally validated with decord to ensure DataLoader compatibility
   - Corrupted clips are detected and can be rebuilt

3. **Potential Data Loss:**
   - Missing source videos → clip not created
   - Invalid timestamps → clip not created or extended to minimum duration
   - Encoding failures → clip marked as failed
   - Validation failures → clip marked as invalid

---

## Stage 3: SAM3 Mask Generation

**Source Script:** `sailsprep/feature_processing/tracker/sam3/run_sam3_batch.py`  
**Batch Script:** `sails_sbatch/tracker/sam3_track.sh`  
**Input CSV:** `sailsprep/subset_data/RMM.csv`  
**Metadata File:** `actreg/dataprep/video_meta.json` (contains video info and cache paths)  
**Output Location:** `/orcd/scratch/bcs/001/sensein/sails/cache_for_tracking/masks/`

### Overview

The SAM3 mask generation process tracks and segments people in full-length videos using SAM3 (Segment Anything Model 3). Mask caches are stored as HDF5 files containing per-frame masks, bounding boxes, scores, and object IDs for downstream processing.

**Note:** Video metadata (FPS, duration, cache paths, frame counts) is stored in `video_meta.json`, which is used for analysis instead of reading HDF5 files directly.

### Process Flow

```
RMM.csv (video list)
    ↓
Read SourceFile and FileName columns
    ↓
Resolve video paths (standardized .mp4 files)
    ↓
Load video frames (capped at 5400 frames)
    ↓
Run SAM3 tracking with "person" prompt
    ↓
Extract per-frame masks, boxes, scores, object IDs
    ↓
Save to HDF5 cache file
    ↓
Update processing_progress.json
    ↓
Generate video_meta.json (with cache paths, fps, duration, cache_frames)
```

### Key Processing Steps

1. **CSV Reading:**
   - Reads `RMM.csv` with video metadata
   - Resolves video paths using `SourceFile` and `FileName` columns
   - Base video directory: `/orcd/data/satra/002/datasets/SAILS/Phase_III_Videos/Videos_from_external_standardized`

2. **Video Processing:**
   - Loads video frames using transformers `load_video`
   - Applies frame stride (default: 1, processes all frames)
   - Caps at `--max-frames 5400` frames per video
   - Processes frames in batches to manage memory

3. **SAM3 Tracking:**
   - Uses Facebook SAM3 model (`facebook/sam3`)
   - Text prompt: "person" (tracks people in videos)
   - Generates per-frame:
     - Object IDs (tracked person instances)
     - Bounding boxes
     - Segmentation masks (RLE encoded)
     - Confidence scores

4. **Mask Cache Storage:**
   - Saves to HDF5 format: `<cache_base>/masks/<video_basename>/facebook-sam3__prompt-person.h5`
   - Video basename: `{filename_stem}_segmented` (derived from output video path)
   - Stores per-frame data in groups: `frame_{index}/obj_ids`, `frame_{index}/boxes`, etc.
   - Includes metadata: height, width, frame_stride, max_frames, model, prompt

5. **Progress Tracking:**
   - Saves `processing_progress.json` with completed video stems
   - Allows resuming interrupted processing
   - Tracks completion status per video

### Output Structure

```
cache_for_tracking/masks/
├── {video_basename}_segmented/
│   └── facebook-sam3__prompt-person.h5
├── ...
```

Each HDF5 file contains:
- **Frame groups:** `frame_{index}/` for each processed frame
  - `obj_ids`: Object IDs (numpy array)
  - `boxes`: Bounding boxes (numpy array, shape: [N, 4])
  - `scores`: Confidence scores (numpy array)
  - `rles`: Run-length encoded masks (variable-length arrays)
- **Attributes:**
  - `height`, `width`: Video dimensions
  - `frame_stride`: Frame sampling stride
  - `max_frames`: Maximum frames processed (5400)
  - `model`: Model name ("facebook-sam3")
  - `prompt`: Text prompt ("person")

### Metadata File: video_meta.json

The `video_meta.json` file contains comprehensive metadata for each video:

- **Video Properties:** `FileName`, `SourceFile`, `original_path`, `width`, `height`, `fps`, `duration`, `rotation_meta`
- **Cache Information:** `mask_cache_path`, `cache_width`, `cache_height`, `cache_stride`, `cache_frames`
- **Analysis Benefits:**
  - No need to read HDF5 files to get frame counts (use `cache_frames`)
  - No need to probe videos for FPS/duration (already in metadata)
  - Direct cache path lookup (no need to derive from filename)
  - Faster analysis (no file I/O for each video)

### Data Statistics

| Metric | Value |
|--------|-------|
| Total videos in metadata | 374 |
| Videos with mask cache paths | ~373 (from metadata) |
| Videos with cache files existing | ~370 (verified on filesystem) |
| Videos marked completed | ~284 (from progress file) |
| Total mask cache files (on filesystem) | 459 |
| Max frames cap | 5400 |
| Frame stride | 1 (all frames) |
| Videos at 5400 frame cap | ~7 (1.5%) |

### Data Flow Notes

1. **Video Path Resolution:**
   - Videos are resolved from CSV `SourceFile` and `FileName` columns
   - Paths are normalized to standardized .mp4 files
   - Mask cache basename: `{filename_stem}_segmented`
   - Cache paths stored in `video_meta.json` for direct lookup

2. **Frame Limiting:**
   - Videos longer than 5400 frames are truncated
   - Only first 5400 frames are processed and cached
   - Frame count stored in `cache_frames` field of metadata
   - This affects downstream processing (clips from later in video may not have masks)
   - Analysis checks if clips start after frame 5400 timestamp

3. **Analysis Using Metadata:**
   - Frame counts read from `cache_frames` (no HDF5 file reading needed)
   - FPS and duration from metadata (no video probing needed)
   - Cache paths directly from `mask_cache_path` field
   - Faster and more reliable than analyzing files individually

4. **Potential Data Loss:**
   - Videos that fail to load → no mask cache
   - Videos that crash during processing → incomplete cache
   - Videos longer than 5400 frames → truncated (frames after 5400 not cached)
   - Processing errors → video skipped, not added to progress
   - Clips starting after frame 5400 timestamp → no SAM3 mask data available
   - **Known Issue:** Video `20211027_212128.mp4` has a clip (`D4Y7P4G2V4_36_month_296_0_3`) that starts at 100.0s, which is after the 5400 frame cap (89.97s at 60.018636 fps). This clip will not have SAM3 mask data available.

5. **Duplicate Filename Handling:**
   - Some videos may have duplicate filenames (e.g., `IMG_5399.MOV` vs `IMG_5399_AM_segmented`)
   - Resolved by renaming cache location in `video_meta.json` and updating the cache folder name
   - Downstream processing relies on `video_meta.json` as the source of truth for cache paths
   - Example: `IMG_5399.MOV` cache renamed from `IMG_5399_segmented` to `IMG_5399_AM_segmented` to avoid conflicts

6. **Progress Tracking:**
   - Progress file tracks completed videos by filename stem
   - Some videos may have caches but not be in progress file (or vice versa)
   - Multiple progress files exist for different batch runs
   - Metadata file provides authoritative source for cache existence

### Known Issues and Resolutions

1. **5400 Frame Cap Impact on Clips:**
   - **Issue:** Video `20211027_212128.mp4` has a clip (`D4Y7P4G2V4_36_month_296_0_3`) that starts at 100.0s, which is after the 5400 frame cap timestamp (89.97s at 60.018636 fps).
   - **Impact:** This clip will not have SAM3 mask data available for downstream processing.
   - **Status:** Being addressed by reprocessing or adjusting clip boundaries.

2. **Duplicate Filename Resolution:**
   - **Issue:** Some videos had duplicate filenames causing cache path conflicts (e.g., `IMG_5399.MOV` vs `IMG_5399_AM_segmented`).
   - **Resolution:** Fixed by renaming the cache location in `video_meta.json` and updating the corresponding cache folder name on disk.
   - **Rationale:** Downstream processing relies on `video_meta.json` as the authoritative source for cache paths, so updating the metadata file ensures consistency.
   - **Note:** This approach works because all downstream code reads cache paths from `video_meta.json` rather than deriving them from filenames.

3. **Videos Excluded from Training/Testing:**
   - **Issue:** Several videos are too high resolution or too long to be processed in full within time constraints. These videos either hit the 5400-frame cap or require significantly more frames than cached (e.g., one video needs ~31k frames but only has 5.4k cached).
   - **Affected Videos:** See `actreg/dataprep/data_review/affected_videos.csv` for the complete list of affected clips and their source videos.
   - **Addiitonal affected vids / clips**:
      - Clips: L4M1H7J7G3_36_month_315_0_0, H7B1M3P5D9_36_month_74_0_4, E2P5Q8S5A7_unknown_319_0_0, J3J0V4T8C3_14_month_52_0_0
      - Vids: 07-02-2019 (2).mov, received_818969921.mp4
   - **Impact:** Clips from these videos will be excluded from model training and testing due to missing or incomplete SAM3 mask cache data.
   - **Status:** Will not be fixed due to time constraints. These videos require full reprocessing with higher frame limits or resolution downsampling, which is not feasible at this time.
   - **Affected Clips Summary:**
     - 7 source videos affected
     - ~18 clips total across these videos
     - Videos include high-resolution content (1920x1080) and/or very long durations (up to ~1037 seconds)
     - Some videos need 10k+ frames but only have 5400 frames cached

### Next Steps

- [ ] Document Stage 2c: HRNet Keypoint Extraction and PKL Generation
- [ ] Document Stage 3: Data Validation (model training dropoffs)


