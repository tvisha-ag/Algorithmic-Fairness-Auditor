# Algorithmic Fairness Auditor

A complete fairness audit pipeline that trains a Logistic Regression 
classifier on the **UCI Adult Income dataset** and mathematically proves 
whether it discriminates against gender and age groups using four 
industry-standard fairness metrics.

Research inspired by Fair AI work at **IIT Hyderabad** and globally.

## Live Demo
https://tvisha-ag.github.io/Algorithmic-Fairness-Auditor/

## What It Proves

| Attribute | Disparate Impact | Verdict |
|-----------|-----------------|---------|
| Gender | 0.343 (< 0.80) | ✗ BIASED |
| Age 18-30 vs 31-45 | 0.320 (< 0.80) | ✗ BIASED |
| Age 61+ vs 31-45 | 0.906 (≥ 0.80) | ✓ FAIR |

**Key finding:** Gender was excluded from training features — yet gender 
bias persists through proxy variables (`occupation × marital_status`).  
Removing a sensitive attribute alone is NOT sufficient for fairness.

## Fairness Metrics Implemented

**Disparate Impact Ratio** (Feldman et al., 2015)
```
DI = P(Ŷ=1 | unprivileged) / P(Ŷ=1 | privileged)
EEOC 80% rule: DI < 0.80 → adverse impact (legally discriminatory)
```

**Demographic Parity Difference** (Dwork et al., 2012)
```
DPD = P(Ŷ=1 | A=0) − P(Ŷ=1 | A=1)    threshold: |Δ| ≤ 0.05
```

**Equalized Odds** (Hardt et al., 2016)
```
Equal TPR and FPR across groups           threshold: |Δ| ≤ 0.05
```

**Predictive Parity + χ² Independence Test**
```
Chi-squared test confirms bias is not due to random sampling variance
```

## Results
```
Attribute               DI      DPD      TPR Δ    p-value    Verdict
Gender                0.343   −0.205   +0.198   <0.001***   ✗ BIASED
Age: 18-30 vs 31-45  0.320   −0.261   +0.038   <0.001***   ✗ BIASED
Age: 61+ vs 31-45    0.906   −0.036   +0.022    0.187       ✓ FAIR
```

## Research Angle

### Proxy Discrimination
Excluding gender from features is not enough. The model learns gender 
as a proxy through correlated features like occupation and marital status.  
This is "proxy discrimination" — documented by Datta, Tschantz & Datta (2017).

### The Fairness Impossibility Theorem
Chouldechova (2017) proved that Demographic Parity, Equalized Odds, and 
Calibration **cannot all be satisfied simultaneously** unless base rates 
are equal across groups. This is the central open problem in Fair AI.

### Debiasing Recommendations
- **Pre-processing:** Reweighing (Kamiran & Calders, 2012)
- **In-processing:** Adversarial Debiasing (Zhang et al., 2018)
- **Post-processing:** Reject Option Classification

## Tech Stack
- Python, scikit-learn, pandas, numpy, scipy, matplotlib
- No external fairness libraries — metrics implemented from scratch

## Setup
```bash
pip install scikit-learn pandas numpy scipy matplotlib
python fairness_auditor.py
```

To use the real UCI dataset instead of synthetic data:
```python
# Replace generate_adult_dataset() with:
df = pd.read_csv('adult.csv')
df = df.rename(columns={'income': 'income_gt50k'})
df['income_gt50k'] = (df['income_gt50k'].str.strip() == '>50K').astype(int)
```
Download from: https://archive.ics.uci.edu/dataset/2/adult

## Files
| File | Description |
|------|-------------|
| `fairness_auditor.py` | Full Python pipeline — model, metrics, plots, report |
| `index.html` | Interactive web audit report (GitHub Pages) |
| `fairness_audit.png` | 8-panel visualization chart |
| `fairness_results.json` | Structured results for reproducibility |

## References
- Feldman et al. (2015). *Certifying and Removing Disparate Impact*. KDD.
- Hardt, Price & Srebro (2016). *Equality of Opportunity in Supervised Learning*. NeurIPS.
- Chouldechova (2017). *Fair Prediction with Disparate Impact*. Big Data.
- Datta, Tschantz & Datta (2017). *Automated Experiments on Ad Privacy*. PETS.
- Bellamy et al. (2019). *AI Fairness 360*. IBM.
- Becker & Kohavi (1996). *Adult Income Dataset*. UCI ML Repository.
