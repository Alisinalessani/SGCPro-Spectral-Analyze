# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Rule-Boosted SGC-Pro (No-ML): Redshift, Line Ratios, Robust Rules + Full Evaluation on SDSS

- Auto-discovery of SDSS spectra with valid subclasses
- Robust preprocessing (continuum removal via ndimage.median_filter, Savitzky–Golay smoothing)
- Redshift estimation by aligning emission/absorption priors
- Line-strength & ratio measurement (OIII/Hb, NII/Ha, SII/Ha, Ha/Hb)
- Rule-based classification (AGN / Star-forming / G-star / K-star / Nebula)
- SGC code generation from emission-peak spacings
- Evaluation against SDSS mapped ground-truth (no sklearn)
- Outputs CSVs and a small summary .txt

Usage (default small sample):
    python main.py
Custom run:
    python main.py --max-plates 6 --fibers-per-plate 30 --tau-samples 120
"""

import argparse
import warnings
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.ndimage import median_filter as nd_median_filter
from astroquery.sdss import SDSS
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Default CONFIG (can be overridden via CLI)
# =========================
MAX_PLATES = 6
FIBERS_PER_PLATE = 30
GLOBAL_TAU_SAMPLE = 120

# preprocessing
WAVE_MIN, WAVE_MAX = 3700.0, 7500.0
SMOOTH_WIN = 21   # must be odd
SMOOTH_POLY = 3
CONT_MED_WIN = 201  # rolling median for continuum (odd)

# peak/trough detection
PEAK_PROM = 0.018
PEAK_DIST = 5
TOL_Z_GRID = 0.0005
Z_MIN, Z_MAX = -0.005, 0.40
LINE_TOL = 4.5

PAD_LEN = 80  # SGC code length

# =========================
# Line sets
# =========================
EMISSION_PRIOR = [
    ("Halpha", 6563, 2.0),
    ("[N II] strong", 6583, 1.5),
    ("[N II]", 6548, 1.0),
    ("[S II]", 6716, 1.2),
    ("[S II] strong", 6731, 1.2),
    ("[O I]", 6300, 1.0),
    ("Hbeta", 4861, 1.8),
    ("[O III] strong", 5007, 2.2),
    ("[O III]", 4959, 1.2),
    ("[O II]", 3727, 1.2),
    ("He II", 4686, 1.0),
    ("[Ne III]", 3869, 1.0),
    ("[Ar III]", 7135, 0.8),
]
ABSORPTION_PRIOR = [
    ("Ca II K", 3933, 1.8),
    ("Ca II H", 3968, 1.6),
    ("Na I D1", 5890, 1.4),
    ("Na I D2", 5896, 1.4),
    ("Mg I b1", 5167, 1.4),
    ("Mg I b2", 5172, 1.4),
    ("Mg I b3", 5183, 1.4),
    ("Fe I", 4383, 0.9),
    ("Ti II", 4550, 0.7),
    ("CN band", 4216, 1.0),
]
KNOWN_LINES = {n: w for (n, w, _w) in EMISSION_PRIOR}
KNOWN_LINES.update({n: w for (n, w, _w) in ABSORPTION_PRIOR})

# =========================
# Utilities
# =========================
def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    """
    Robust median filter using scipy.ndimage.median_filter to avoid dtype issues.
    """
    if win % 2 == 0:
        win += 1
    if win < 5 or win > len(x):
        return x
    x = np.asarray(x)
    try:
        return nd_median_filter(x.astype(np.float64, copy=False), size=int(win), mode='nearest')
    except Exception:
        return x

def preprocess(lam: np.ndarray, flux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lam = np.asarray(lam); flux = np.asarray(flux)
    mask = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
    lam, flux = lam[mask], flux[mask]
    if lam.size == 0:
        return lam, flux
    # crop wavelength window
    m2 = (lam >= WAVE_MIN) & (lam <= WAVE_MAX)
    lam, flux = lam[m2], flux[m2]
    if lam.size == 0:
        return lam, flux
    # continuum removal (divide by rolling median), then smooth
    cont = rolling_median(flux, CONT_MED_WIN)
    med_flux = np.median(flux) if np.isfinite(np.median(flux)) else 1.0
    cont[~np.isfinite(cont) | (cont <= 0)] = med_flux
    norm = flux / cont
    win = SMOOTH_WIN if (SMOOTH_WIN % 2 == 1) else (SMOOTH_WIN + 1)
    if len(norm) >= win:
        try:
            norm = sig.savgol_filter(norm, win, SMOOTH_POLY)
        except Exception:
            pass
    return lam, norm

def find_peaks_troughs(y: np.ndarray):
    p_idx, p_props = sig.find_peaks(y, prominence=PEAK_PROM, distance=PEAK_DIST)
    t_idx, t_props = sig.find_peaks(-y, prominence=PEAK_PROM/1.2, distance=PEAK_DIST)
    return (p_idx, p_props), (t_idx, t_props)

def fetch_spectrum(plate: int, mjd: int, fiber: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        sp = SDSS.get_spectra(plate=plate, mjd=mjd, fiberID=fiber)
        if not sp:
            return None, None
        hdu = sp[0][1]
        lam = 10.0 ** hdu.data["loglam"]
        flux = hdu.data["flux"]
        return lam, flux
    except Exception:
        return None, None

def estimate_z(lam: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
    (p_idx, p_props), (t_idx, t_props) = find_peaks_troughs(y)
    if p_idx.size + t_idx.size < 2:
        return 0.0, {}

    peaks_wave = lam[p_idx]; peaks_prom = p_props.get("prominences", np.ones_like(peaks_wave))
    trough_wave = lam[t_idx]; trough_prom = t_props.get("prominences", np.ones_like(trough_wave))

    obs_lines = np.concatenate([peaks_wave, trough_wave])
    obs_prom  = np.concatenate([peaks_prom, trough_prom])

    z_grid = np.arange(Z_MIN, Z_MAX + TOL_Z_GRID, TOL_Z_GRID)
    best_score, best_z = -1.0, 0.0
    best_matches = {}
    for z in z_grid:
        score = 0.0
        matches = {}
        for (name, w0, wght) in EMISSION_PRIOR:
            w_obs = w0 * (1.0 + z)
            j = np.argmin(np.abs(obs_lines - w_obs))
            if np.abs(obs_lines[j] - w_obs) <= LINE_TOL:
                score += wght * (1.0 + obs_prom[j])
                matches[name] = (w0, w_obs, obs_lines[j], obs_prom[j], "em_or_abs")
        for (name, w0, wght) in ABSORPTION_PRIOR:
            w_obs = w0 * (1.0 + z)
            j = np.argmin(np.abs(obs_lines - w_obs))
            if np.abs(obs_lines[j] - w_obs) <= LINE_TOL:
                score += 0.8 * wght * (1.0 + obs_prom[j])
                matches[name] = (w0, w_obs, obs_lines[j], obs_prom[j], "em_or_abs")
        if score > best_score:
            best_score, best_z, best_matches = score, z, matches
    return float(best_z), best_matches

def measure_line_strengths(lam: np.ndarray, y: np.ndarray, z_est: float) -> Dict[str, float]:
    lam_rest = lam / (1.0 + z_est)
    (p_idx, p_props), (t_idx, t_props) = find_peaks_troughs(y)
    peaks = {"wave": lam_rest[p_idx], "prom": p_props.get("prominences", np.ones_like(p_idx))}
    troughs = {"wave": lam_rest[t_idx], "prom": t_props.get("prominences", np.ones_like(t_idx))}

    strengths: Dict[str, float] = {}

    def nearest_feature(w0: float, is_emission: bool):
        waves = peaks["wave"] if is_emission else troughs["wave"]
        prom  = peaks["prom"] if is_emission else troughs["prom"]
        if waves.size == 0:
            return None
        j = np.argmin(np.abs(waves - w0))
        if np.abs(waves[j] - w0) <= LINE_TOL:
            return float(prom[j]), float(abs(waves[j] - w0))
        return None

    for (name, w0, _w) in EMISSION_PRIOR:
        hit = nearest_feature(w0, True)
        strengths[name] = hit[0] if hit is not None else 0.0
    for (name, w0, _w) in ABSORPTION_PRIOR:
        hit = nearest_feature(w0, False)
        strengths[name] = hit[0] if hit is not None else 0.0
    return strengths

def classify_rules(stren: Dict[str, float]) -> Tuple[str, Dict]:
    eps = 1e-8
    o3 = stren.get("[O III] strong", 0.0)
    hb = stren.get("Hbeta", 0.0)
    ha = stren.get("Halpha", 0.0)
    n2 = stren.get("[N II] strong", 0.0)
    s2 = max(stren.get("[S II]", 0.0), stren.get("[S II] strong", 0.0))

    r_o3_hb = o3 / (hb + eps)
    r_n2_ha = n2 / (ha + eps)
    r_s2_ha = s2 / (ha + eps)

    ca = max(stren.get("Ca II K",0.0), stren.get("Ca II H",0.0))
    na = max(stren.get("Na I D1",0.0), stren.get("Na I D2",0.0))
    mgb = max(stren.get("Mg I b1",0.0), stren.get("Mg I b2",0.0), stren.get("Mg I b3",0.0))
    cn = stren.get("CN band", 0.0)

    em_count = sum(1 for k in ["Halpha","Hbeta","[O III] strong","[O III]","[O II]","[N II] strong","[S II]"] if stren.get(k,0.0) > 0)
    abs_count = sum(1 for k in ["Ca II K","Ca II H","Na I D1","Na I D2","Mg I b1","Mg I b2","Mg I b3","CN band"] if stren.get(k,0.0) > 0)

    if em_count >= 2 and ha + hb + o3 > 0.05:
        if (r_o3_hb >= 3.0) or (r_n2_ha >= 0.6) or (r_s2_ha >= 0.5):
            return "AGN Galaxy", {"r_OIII_Hb": r_o3_hb, "r_NII_Ha": r_n2_ha, "r_SII_Ha": r_s2_ha, "reason": "High line ratios (AGN-like)."}
        if (ha > 0 and hb > 0) and (o3 > 0) and (r_n2_ha < 0.6) and (r_s2_ha < 0.5):
            return "Star-forming Galaxy", {"r_OIII_Hb": r_o3_hb, "r_NII_Ha": r_n2_ha, "r_SII_Ha": r_s2_ha, "reason": "Balmer + [OIII] with low [NII]/[SII]."}
        if (o3 > 0.6 or s2 > 0.6) and (abs_count <= 1):
            return "Nebula (HII or PN)", {"r_OIII_Hb": r_o3_hb, "r_NII_Ha": r_n2_ha, "r_SII_Ha": r_s2_ha, "reason": "Strong forbidden lines, weak stellar absorption."}

    if abs_count >= 2 and em_count <= 1:
        if (ca > 0.5 and na > 0.4 and mgb > 0.4):
            return "K-type Star", {"CA": ca, "NA": na, "MGB": mgb, "reason": "Strong Ca II/Na D/Mg b absorption."}
        if (ca > 0.3 and (cn > 0.25 or mgb > 0.25) and na < 0.6):
            return "G-type Star", {"CA": ca, "CN": cn, "MGB": mgb, "NA": na, "reason": "Ca II + CN/Mg b, moderate Na D."}

    if em_count >= 1 and abs_count == 0:
        return "Star-forming Galaxy", {"reason": "Emission features present."}
    return "Uncertain", {"reason": "Mixed or weak features."}

def explain_label(lbl: str, extras: Dict) -> str:
    if lbl == "AGN Galaxy":
        return "High [O III]/Hbeta or [N II]/Halpha or [S II]/Halpha → AGN-like excitation."
    if lbl == "Star-forming Galaxy":
        return "Balmer + [O III] with low [N II]/[S II] → HII/star-forming signatures."
    if lbl == "K-type Star":
        return "Strong Ca II H&K, Na D, Mg b absorptions → K-type stellar spectrum."
    if lbl == "G-type Star":
        return "Ca II + CN/Mg b with moderate Na D → G-type stellar spectrum."
    if lbl == "Nebula (HII or PN)":
        return "Strong [O III]/[S II], weak stellar absorption → nebular emission."
    return extras.get("reason","Unclear.")

def encode_sgc_from_rest(lam_rest: np.ndarray, y_rest: np.ndarray, tau1: float, tau2: float, pad_length: int = PAD_LEN):
    try:
        p_idx, _ = sig.find_peaks(y_rest, prominence=PEAK_PROM, distance=PEAK_DIST)
    except Exception:
        return "No peaks", np.array([])
    peak_waves = lam_rest[p_idx]
    if peak_waves.size < 2:
        return "Too few peaks", peak_waves
    dlambda = np.diff(peak_waves)
    code = "".join("0" if d < tau1 else "1" if d < tau2 else "2" for d in dlambda)
    code = code[:pad_length].ljust(pad_length, "9")
    return code, peak_waves

# =========================
# Ground-truth mapping
# =========================
def map_sdss_to_label(sdss_class: str, sdss_subclass: str) -> str:
    c = (sdss_class or "").strip().upper()
    s = (sdss_subclass or "").strip().upper()
    if c in ["GALAXY","QSO"] and any(k in s for k in ["AGN","SEYFERT","LINER","BROADLINE","QSO"]):
        return "AGN Galaxy"
    if c == "GALAXY" and any(k in s for k in ["STARFORMING","HII","EMISSION"]):
        return "Star-forming Galaxy"
    if c == "STAR":
        if "K" in s and not any(x in s for x in ["WK","MK"]):
            return "K-type Star"
        if "G" in s:
            return "G-type Star"
    if any(k in s for k in ["PN","PLANETARY","HII","NEBULA"]):
        return "Nebula (HII or PN)"
    if c == "STAR":
        return "G-type Star" if "G" in s else "K-type Star" if "K" in s else "Uncertain"
    if c in ["GALAXY","QSO"]:
        return "Uncertain"
    return "Uncertain"

# =========================
# Discover labeled samples from SDSS
# =========================
def discover_labeled_samples(max_plates=MAX_PLATES, fibers_per_plate=FIBERS_PER_PLATE):
    sql = """
    SELECT TOP 6000 plate, mjd, fiberid, class, subclass
    FROM specObj
    WHERE subclass IS NOT NULL AND subclass <> ''
      AND class IN ('GALAXY','STAR','QSO')
    ORDER BY plate, mjd
    """
    tbl = SDSS.query_sql(sql)
    if tbl is None or len(tbl) == 0:
        return []
    df = tbl.to_pandas()
    df["GT_Label"] = [map_sdss_to_label(c, s) for c, s in zip(df["class"], df["subclass"])]
    df = df[df["GT_Label"] != "Uncertain"]
    groups = df.groupby(["plate","mjd"])
    selected = []
    unique_pm = set()
    for (pl, mj), g in groups:
        g = g.head(fibers_per_plate)
        for _, r in g.iterrows():
            selected.append((int(r["plate"]), int(r["mjd"]), int(r["fiberid"]), str(r["class"]), str(r["subclass"]), r["GT_Label"]))
        unique_pm.add((int(pl), int(mj)))
        if len(unique_pm) >= max_plates:
            break
    return selected

# =========================
# Estimate global taus
# =========================
def estimate_global_taus(samples, max_samples=GLOBAL_TAU_SAMPLE):
    spacings: List[float] = []
    for (pl, mj, fiber, *_t) in tqdm(samples[:max_samples], desc="Estimating global taus"):
        lam, flux = fetch_spectrum(pl, mj, fiber)
        if lam is None:
            continue
        lam, y = preprocess(lam, flux)
        if lam.size == 0:
            continue
        z_est, _m = estimate_z(lam, y)
        lam_rest = lam / (1.0 + z_est)
        try:
            p_idx, _ = sig.find_peaks(y, prominence=PEAK_PROM, distance=PEAK_DIST)
        except Exception:
            continue
        pw = lam_rest[p_idx]
        pw = pw[(pw >= WAVE_MIN) & (pw <= WAVE_MAX)]
        if pw.size < 2:
            continue
        dl = np.diff(pw)
        if np.all(np.isfinite(dl)) and dl.size > 0:
            spacings.extend(dl.tolist())
        time.sleep(0.01)
    if len(spacings) == 0:
        return 20.0, 35.0
    spacings = np.array(spacings)
    tau1 = float(np.median(spacings))
    tau2 = float(np.percentile(spacings, 80))
    if tau2 < tau1:
        tau2 = tau1 * 1.25
    return tau1, tau2

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Rule-Boosted SGC-Pro (No-ML) for SDSS spectra.")
    parser.add_argument("--max-plates", type=int, default=MAX_PLATES, help="Number of distinct plate/mjd groups to sample.")
    parser.add_argument("--fibers-per-plate", type=int, default=FIBERS_PER_PLATE, help="Fibers per plate to fetch.")
    parser.add_argument("--tau-samples", type=int, default=GLOBAL_TAU_SAMPLE, help="Max samples used to estimate global taus.")
    parser.add_argument("--wave-min", type=float, default=WAVE_MIN)
    parser.add_argument("--wave-max", type=float, default=WAVE_MAX)
    args = parser.parse_args()

    global MAX_PLATES, FIBERS_PER_PLATE, GLOBAL_TAU_SAMPLE, WAVE_MIN, WAVE_MAX
    MAX_PLATES, FIBERS_PER_PLATE, GLOBAL_TAU_SAMPLE = args.max_plates, args.fibers_per_plate, args.tau_samples
    WAVE_MIN, WAVE_MAX = args.wave_min, args.wave_max

    print("Discovering labeled SDSS samples ...")
    labeled = discover_labeled_samples(MAX_PLATES, FIBERS_PER_PLATE)
    if len(labeled) == 0:
        raise RuntimeError("No labeled samples found. Try increasing --max-plates / --fibers-per-plate.")

    unique_pm_count = len(set((pl, mj) for (pl, mj, *_rest) in labeled))
    print(f"Found {len(labeled)} fibers across {unique_pm_count} plate/mjd with valid ground-truth.")

    tau1, tau2 = estimate_global_taus(labeled, max_samples=GLOBAL_TAU_SAMPLE)
    print(f"\nGlobal taus: tau1={tau1:.3f}, tau2={tau2:.3f}\n")

    rows = []
    eval_rows = []
    not_found = 0

    for (pl, mj, fiber, sdss_class, sdss_subclass, gt_label) in tqdm(labeled, desc="Processing labeled spectra"):
        lam, flux = fetch_spectrum(pl, mj, fiber)
        if lam is None:
            not_found += 1
            rows.append({
                "Plate": pl, "MJD": mj, "Fiber": fiber,
                "z_est": np.nan,
                "Halpha/Hbeta": np.nan,
                "[OIII]/Hbeta": np.nan,
                "[NII]/Halpha": np.nan,
                "[SII]/Halpha": np.nan,
                "Detected_Elements": "",
                "SGC_Code": "No spectrum",
                "Prediction": "Uncertain",
                "Explanation": "Spectrum not found.",
                "SDSS_Class": sdss_class,
                "SDSS_Subclass": sdss_subclass,
                "GT_Label": gt_label
            })
            continue

        lam, y = preprocess(lam, flux)
        if lam.size == 0:
            continue

        z_est, matches = estimate_z(lam, y)
        strengths = measure_line_strengths(lam, y, z_est)
        eps = 1e-8
        ha = strengths.get("Halpha",0.0); hb = strengths.get("Hbeta",0.0)
        o3 = strengths.get("[O III] strong",0.0)
        n2s = strengths.get("[N II] strong",0.0)
        s2m = max(strengths.get("[S II]",0.0), strengths.get("[S II] strong",0.0))

        r_ha_hb = ha/(hb+eps) if ha>0 and hb>0 else np.nan
        r_o3_hb = o3/(hb+eps) if hb>0 else np.nan
        r_n2_ha = n2s/(ha+eps) if ha>0 else np.nan
        r_s2_ha = s2m/(ha+eps) if ha>0 else np.nan

        pred, extras = classify_rules(strengths)
        explanation = explain_label(pred, extras)

        lam_rest = lam / (1.0 + z_est if (1.0 + z_est) != 0 else 1.0)
        sgc_code, _pw = encode_sgc_from_rest(lam_rest, y, tau1, tau2, pad_length=PAD_LEN)

        detected_elements = []
        for name, w0 in KNOWN_LINES.items():
            val = strengths.get(name,0.0)
            if val > 0:
                detected_elements.append(name)
        detected_elements = ", ".join(sorted(set(detected_elements)))

        rows.append({
            "Plate": pl, "MJD": mj, "Fiber": fiber,
            "z_est": z_est,
            "Halpha/Hbeta": r_ha_hb,
            "[OIII]/Hbeta": r_o3_hb,
            "[NII]/Halpha": r_n2_ha,
            "[SII]/Halpha": r_s2_ha,
            "Detected_Elements": detected_elements,
            "SGC_Code": sgc_code,
            "Prediction": pred,
            "Explanation": explanation,
            "SDSS_Class": sdss_class,
            "SDSS_Subclass": sdss_subclass,
            "GT_Label": gt_label
        })

        if pred != "Uncertain" and gt_label != "Uncertain":
            eval_rows.append({"pred": pred, "gt": gt_label})

    # save results
    df = pd.DataFrame(rows)
    df.to_csv("SGC_SDSS_RuleBoosted_Results.csv", index=False)
    print("Saved per-fiber results: SGC_SDSS_RuleBoosted_Results.csv")
    print(f"Spectra not found: {not_found}")

    # evaluation
    if len(eval_rows) == 0:
        print("\nNo evaluable samples (no mapped ground-truth). Consider increasing --max-plates/--fibers-per-plate.")
    else:
        eval_df = pd.DataFrame(eval_rows)
        total = len(eval_df)
        correct = int((eval_df["pred"] == eval_df["gt"]).sum())
        accuracy = correct / total if total > 0 else 0.0

        cm = pd.crosstab(eval_df["gt"], eval_df["pred"], rownames=["GT"], colnames=["Pred"], dropna=False)
        labels = sorted(set(eval_df["gt"]).union(set(eval_df["pred"])))

        def prf(lbl: str):
            tp = int(((eval_df["pred"] == lbl) & (eval_df["gt"] == lbl)).sum())
            fp = int(((eval_df["pred"] == lbl) & (eval_df["gt"] != lbl)).sum())
            fn = int(((eval_df["pred"] != lbl) & (eval_df["gt"] == lbl)).sum())
            precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
            recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            support   = int((eval_df["gt"] == lbl).sum())
            return precision, recall, f1, support

        per_class = []
        P, R, F1 = [], [], []
        for lbl in labels:
            p, r, f, s = prf(lbl)
            per_class.append({"Class": lbl, "Precision": p, "Recall": r, "F1": f, "Support": s})
            P.append(p); R.append(r); F1.append(f)

        macro_p = float(np.mean(P)) if P else 0.0
        macro_r = float(np.mean(R)) if R else 0.0
        macro_f1 = float(np.mean(F1)) if F1 else 0.0
        micro_f1 = accuracy

        print("\n===== Evaluation (Rule-Boosted; valid GT & non-Uncertain preds) =====")
        print(f"Samples: {total} | Correct: {correct} | Accuracy: {accuracy:.3f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nPer-class metrics:")
        print(pd.DataFrame(per_class).to_string(index=False))
        print(f"\nMacro Precision: {macro_p:.3f} | Macro Recall: {macro_r:.3f} | Macro F1: {macro_f1:.3f}")
        print(f"Micro Precision/Recall/F1 (== Accuracy): {micro_f1:.3f}")

        cm.to_csv("SGC_eval_confusion_matrix.csv")
        pd.DataFrame(per_class).to_csv("SGC_eval_per_class.csv", index=False)
        with open("SGC_eval_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Samples: {total}\nCorrect: {correct}\nAccuracy: {accuracy:.6f}\n")
            f.write(f"Macro P/R/F1: {macro_p:.6f}/{macro_r:.6f}/{macro_f1:.6f}\n")
            f.write(f"Micro (acc): {micro_f1:.6f}\n")
        print("\nSaved: SGC_eval_confusion_matrix.csv, SGC_eval_per_class.csv, SGC_eval_summary.txt")

if __name__ == "__main__":
    main()
