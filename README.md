# SGCPro-Spectral-Analyze
Python implementation of Rule-Boosted SGC-Pro for SDSS spectra classification.


# SGCPro-Spectral-Analyze

**Rule-Boosted Spectral Galaxy Classifier (SGC-Pro) for SDSS spectra**  
This project classifies Sloan Digital Sky Survey (SDSS) spectra into **AGN Galaxies, Star-forming Galaxies, Stars (G/K types), or Nebulae** using astrophysical rules and emission-line diagnostics – **without machine learning**.

---

## ✨ Features
- Automatic retrieval of SDSS spectra with valid subclasses  
- Preprocessing: continuum removal & smoothing  
- Redshift estimation via emission/absorption line matching  
- Emission and absorption line measurement  
- Rule-based classification (BPT-like diagnostics)  
- Generation of **SGC codes** from emission peak spacings  
- Evaluation: confusion matrix, precision, recall, F1  

---

## 📦 Requirements
Install required packages:

```bash
pip install -r requirements.txt
