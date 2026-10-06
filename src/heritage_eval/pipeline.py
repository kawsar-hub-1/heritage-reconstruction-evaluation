
"""
Heritage Reconstruction Evaluation Pipeline — v2 (with guidance & reliability stats)
------------------------------------------------------------------------------------
What this does (in plain English):
1) Opens your archival photo (the real site) and your AI images.
2) Aligns each AI image to the archival photo so comparisons are fair.
3) Looks only inside the regions that matter (ROIs you define): statue, robe, niche, etc.
4) Measures similarity/structure INSIDE those regions (not the whole image).
5) Reads your Excel responses; counts expert votes for each image and computes agreement.
6) Compares the algorithm’s score with expert votes and saves clean figures + a CSV.

You will run it TWICE: once for Bamiyan, once for Bel.

Quick folder setup (make these anywhere you like):
data/
  archival/           -> put bamiyan_ref.jpg (and bel_ref.jpg later)
  generated_baseline/ -> put your baseline images numbered e.g., 1.png ... 7.png
results/              -> the script will write charts/CSVs here

Run (example for Bamiyan):
python heritage_eval_pipeline.py \
  --excel "/path/to/AI-Driven Heritage Reconstruction Validation Questionnaire (Responses).xlsx" \
  --archival_dir data/archival \
  --baseline_dir data/generated_baseline \
  --rois rois_example.json \
  --case Bamiyan \
  --out results

ROIs (regions of interest):
- Edit rois_example.json: move the polygon points (x,y) to roughly surround statue, robe, niche.
  If you do nothing, the script uses rough rectangles; that's still OK to start.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr

# -----------------------------
# Config container
# -----------------------------
@dataclass
class Config:
    excel: Path | None
    archival_dir: Path
    baseline_dir: Path
    rois_path: Path
    case: str  # "Bamiyan" or "Bel"
    out_dir: Path

# -----------------------------
# Helper utilities
# -----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def imread_any(path: Path):
    img = cv2.imread(str(path))
    if img is None: raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

# -----------------------------
# Registration
# -----------------------------
def register_to_reference(ref_bgr: np.ndarray, img_bgr: np.ndarray):
    """Register img_bgr to ref_bgr using ORB keypoints + homography; return warped and diagnostics."""
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return img_bgr.copy(), None, [], None  # fail gracefully

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) == 0:
        return img_bgr.copy(), None, [], None
    matches = sorted(matches, key=lambda m: m.distance)[:500]

    dst = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    src = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None: return img_bgr.copy(), None, matches, None

    warped = cv2.warpPerspective(img_bgr, H, (ref_bgr.shape[1], ref_bgr.shape[0]))
    return warped, H, matches, inliers

def plot_registration_demo(ref_bgr, img_bgr, warped_bgr, out_path: Path):
    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(1,3,1); ax1.imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)); ax1.set_title('Archival')
    ax2 = fig.add_subplot(1,3,2); ax2.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax2.set_title('Generated')
    ax3 = fig.add_subplot(1,3,3)
    overlay = cv2.addWeighted(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB), 0.5,
                              cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB), 0.5, 0)
    ax3.imshow(overlay); ax3.set_title('Overlay (after warp)')
    for ax in (ax1,ax2,ax3): ax.axis('off')
    savefig(out_path)

# -----------------------------
# ROIs + metrics
# -----------------------------
def apply_mask(img: np.ndarray, polygon: list[tuple[int,int]]):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    out = cv2.bitwise_and(img, img, mask=mask)
    return out, mask

def ssim_Lab(ref_bgr, gen_bgr, mask=None):
    ref_L = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB)[:,:,0]
    gen_L = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2LAB)[:,:,0]
    if mask is not None:
        ref_L = ref_L * (mask>0); gen_L = gen_L * (mask>0)
    score, _ = ssim(ref_L, gen_L, full=True)
    return float(score)

def proportion_ratio_from_mask(mask: np.ndarray):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return np.nan
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return np.nan if w==0 else float(h)/float(w)

def drapery_entropy(bgr: np.ndarray, mask: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    vals = mag[mask>0]
    if vals.size == 0: return np.nan
    hist, _ = np.histogram(vals, bins=32, density=True)
    hist = hist[hist>0]
    return float(-np.sum(hist*np.log2(hist)))

def column_spacing_ratio(bgr: np.ndarray, mask: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=40, maxLineGap=10)
    if lines is None: return np.nan
    xs = []
    for x1,y1,x2,y2 in lines[:,0]:
        if abs(x2-x1) < 4 and abs(y2-y1) > 20: xs.append((x1+x2)//2)
    xs = sorted(xs)
    if len(xs) < 3: return np.nan
    diffs = np.diff(xs); w = mask.shape[1]
    return float(np.median(diffs))/float(w)

def block_joint_frequency(bgr: np.ndarray, mask: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    count = np.count_nonzero(edges)
    return float(count)/(h*w)

# -----------------------------
# Excel parsing (expert votes + reliability)
# -----------------------------
def parse_votes_column(df: pd.DataFrame, question_keyword: str, image_ids: list[int]):
    cols = [c for c in df.columns if isinstance(c, str) and question_keyword.lower() in c.lower()]
    if not cols: raise ValueError(f"Could not find a column containing: {question_keyword}")
    col = cols[0]
    counts = {i:0 for i in image_ids}

    # Build rater-by-image binary matrix for Fleiss' kappa
    R = []  # each row is a rater's 0/1 selections across images
    for _, row in df.iterrows():
        sel = [0]*len(image_ids)
        val = row.get(col, np.nan)
        if pd.isna(val):
            R.append(sel); continue
        parts = [p.strip() for p in str(val).replace(';',',').split(',')]
        for k, img_id in enumerate(image_ids):
            if any(("image" in p.lower() and str(img_id) in p) for p in parts):
                sel[k] = 1; counts[img_id] += 1
        R.append(sel)
    R = np.array(R, dtype=int)  # shape: raters × images
    return counts, R

def fleiss_kappa_binary(R: np.ndarray):
    """
    R: raters × items matrix with 0/1 (not selected / selected).
    We treat each item (image) as subject with 2 categories.
    Implements Fleiss' kappa for nominal categories.
    """
    if R.ndim != 2: raise ValueError("R must be 2D (raters × items)")
    n_raters = R.shape[0]
    n_items  = R.shape[1]

    # For each item j, count n_selected and n_not
    n_sel = R.sum(axis=0)              # length n_items
    n_not = n_raters - n_sel
    Pj = (n_sel*(n_sel-1) + n_not*(n_not-1)) / (n_raters*(n_raters-1) + 1e-9)  # agreement for item j
    Pbar = Pj.mean()

    # Category proportions across all ratings
    p_sel = n_sel.sum() / (n_items*n_raters)
    p_not = 1 - p_sel
    Pbar_e = p_sel**2 + p_not**2

    kappa = (Pbar - Pbar_e) / (1 - Pbar_e + 1e-9)
    return float(kappa), float(Pbar), float(Pbar_e)

# -----------------------------
# Core pipeline per case
# -----------------------------
def run_case(cfg: Config):
    ensure_dir(cfg.out_dir)
    rois = json.loads(Path(cfg.rois_path).read_text())
    if cfg.case not in rois: raise ValueError(f"Case '{cfg.case}' not found in ROI file")
    case_rois = rois[cfg.case]

    # Archival reference (first image found)
    arch_candidates = sorted([p for p in cfg.archival_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not arch_candidates: raise FileNotFoundError(f"No archival images in {cfg.archival_dir}")
    ref_path = arch_candidates[0]; ref = imread_any(ref_path)

    # Baseline images
    baselines = sorted([p for p in cfg.baseline_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}], key=lambda p: p.name)
    if not baselines: raise FileNotFoundError(f"No baseline images in {cfg.baseline_dir}")

    rows = []; demo_done=False
    for p in baselines:
        img = imread_any(p)
        warped,H,_,_ = register_to_reference(ref, img)
        if not demo_done:
            plot_registration_demo(ref, img, warped, cfg.out_dir/f"{cfg.case.lower()}_registration_overlay.png")
            demo_done=True

        metrics = {"Image": p.stem}
        composite_terms = []
        for roi_name, poly in case_rois.items():
            roi_ref,_ = apply_mask(ref, poly)
            roi_gen,mask = apply_mask(warped, poly)
            ssim_score = ssim_Lab(roi_ref, roi_gen, mask)
            metrics[f"SSIM_{roi_name}"] = ssim_score
            composite_terms.append(ssim_score)

            if cfg.case.lower() == 'bamiyan':
                if 'robe' in roi_name.lower(): metrics['DraperyEntropy'] = drapery_entropy(roi_gen, mask)
                if 'statue' in roi_name.lower(): metrics['PropRatio'] = proportion_ratio_from_mask(mask)
            else:
                if 'columns' in roi_name.lower(): metrics['ColumnSpacingRatio'] = column_spacing_ratio(roi_gen, mask)
                if 'wall' in roi_name.lower() or 'facade' in roi_name.lower(): metrics['JointFreq'] = block_joint_frequency(roi_gen, mask)

        valid = [t for t in composite_terms if not np.isnan(t)]
        metrics['CompositeScore'] = float(np.mean(valid)) if valid else np.nan
        rows.append(metrics)

    df = pd.DataFrame(rows)
    csv_path = cfg.out_dir/f"{cfg.case.lower()}_metrics.csv"; df.to_csv(csv_path, index=False)

    # ----- Excel parsing & agreement -----
    if cfg.excel is not None and cfg.excel.exists():
        excel_df = pd.read_excel(cfg.excel)

        if cfg.case.lower() == 'bamiyan':
            # Q12 structural (Images 1–7) – use exact text you showed me
            q12 = "Which of the above images (1–7) best represent(s) the architectural and structural accuracy of the Buddhas of Bamiyan?"
            counts12, R12 = parse_votes_column(excel_df, q12, list(range(1,8)))
            ids = sorted(counts12.keys()); counts = [counts12[i] for i in ids]
            plt.figure(figsize=(7,4)); plt.bar([f"Image {i}" for i in ids], counts)
            plt.title('Bamiyan — Expert votes: structural accuracy (Q12)'); plt.xlabel('Image'); plt.ylabel('Votes')
            savefig(cfg.out_dir/'bamiyan_votes_Q12.png')

            kappa12, Pbar12, Pe12 = fleiss_kappa_binary(R12)
            with open(cfg.out_dir/'bamiyan_agreement_Q12.txt','w') as f:
                f.write(f"Fleiss' kappa (Q12): {kappa12:.3f}\nPbar (observed agreement): {Pbar12:.3f}\nPe (expected): {Pe12:.3f}\n")

            # Q18 artistic (Images 8–17)
            q18 = "From the artistic details below (Images 8–17), which ones most authentically reflect the Buddha’s style and iconography?"
            counts18, R18 = parse_votes_column(excel_df, q18, list(range(8,18)))
            ids18 = sorted(counts18.keys()); counts18_list = [counts18[i] for i in ids18]
            plt.figure(figsize=(7,4)); plt.bar([f"Image {i}" for i in ids18], counts18_list)
            plt.title('Bamiyan — Expert votes: artistic details (Q18)'); plt.xlabel('Image'); plt.ylabel('Votes')
            savefig(cfg.out_dir/'bamiyan_votes_Q18.png')

            kappa18, Pbar18, Pe18 = fleiss_kappa_binary(R18)
            with open(cfg.out_dir/'bamiyan_agreement_Q18.txt','w') as f:
                f.write(f"Fleiss' kappa (Q18): {kappa18:.3f}\nPbar (observed): {Pbar18:.3f}\nPe (expected): {Pe18:.3f}\n")

            # Correlate algorithm vs votes for Q12 if image IDs match
            try:
                df_num = df.copy()
                df_num['ImageID'] = df_num['Image'].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
                merged = df_num.merge(pd.DataFrame({'ImageID':ids,'Votes':counts}), on='ImageID', how='inner')
                if not merged.empty:
                    rho, p = spearmanr(merged['CompositeScore'], merged['Votes'])
                    plt.figure(figsize=(5,4)); plt.scatter(merged['CompositeScore'], merged['Votes'])
                    for _,r in merged.iterrows():
                        plt.annotate(str(r['ImageID']), (r['CompositeScore'], r['Votes']))
                    plt.title(f'Algorithm vs Expert votes (Q12): ρ={rho:.2f}, p={p:.3f}')
                    plt.xlabel('Composite algorithmic score'); plt.ylabel('Expert votes')
                    savefig(cfg.out_dir/'bamiyan_alg_vs_votes_Q12.png')
            except Exception as e:
                print('Correlation step skipped:', e)

        else:
            # Bel — you must update this keyword once you tell me the exact question text.
            # For now we attempt to find any column containing 'Temple of Bel' and numbers 1–19
            try:
                qbel = "Temple of Bel"
                counts_bel, Rbel = parse_votes_column(excel_df, qbel, list(range(1,20)))
                ids = sorted(counts_bel.keys()); counts = [counts_bel[i] for i in ids]
                plt.figure(figsize=(7,4)); plt.bar([f"Image {i}" for i in ids], counts)
                plt.title('Temple of Bel — Expert votes'); plt.xlabel('Image'); plt.ylabel('Votes')
                savefig(cfg.out_dir/'bel_votes.png')
                kappa_bel, Pbar_bel, Pe_bel = fleiss_kappa_binary(Rbel)
                with open(cfg.out_dir/'bel_agreement.txt','w') as f:
                    f.write(f"Fleiss' kappa (Bel): {kappa_bel:.3f}\nPbar: {Pbar_bel:.3f}\nPe: {Pe_bel:.3f}\n")
            except Exception as e:
                print('Bel parsing/skipped:', e)

    print(f"Saved metrics: {csv_path}")

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run heritage evaluation on Bamiyan or Bel")
    parser.add_argument('--excel', type=str, default=None, help='Path to your Excel responses')
    parser.add_argument('--archival_dir', type=str, required=True)
    parser.add_argument('--baseline_dir', type=str, required=True)
    parser.add_argument('--rois', type=str, required=True)
    parser.add_argument('--case', type=str, choices=['Bamiyan','Bel'], required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    cfg = Config(
        excel=Path(args.excel) if args.excel else None,
        archival_dir=Path(args.archival_dir),
        baseline_dir=Path(args.baseline_dir),
        rois_path=Path(args.rois),
        case=args.case,
        out_dir=ensure_dir(Path(args.out))
    )
    np.random.seed(42)
    run_case(cfg)

if __name__ == '__main__':
    main()
