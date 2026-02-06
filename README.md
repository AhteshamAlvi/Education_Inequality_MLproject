Education Inequality – Student Performance
==========================================

This project explores how socioeconomic background and study habits relate to college performance. Using a publicly available student‑performance dataset, it runs three hypothesis tests, visualizes the results, and trains baseline regression models to predict cumulative college GPA.

Project contents
----------------
- `main.py` – end‑to‑end exploratory analysis, hypothesis tests, visualizations, and two GPA prediction models (Linear Regression and Random Forest).
- `test.py` – a cleaner scikit‑learn pipeline version of the regression experiment (ColumnTransformer + Pipeline).
- `ml_example.py` – small synthetic datasets to illustrate regression behavior; useful for teaching/model intuition.
- `ResearchInformation3.csv` – input data (student demographics, academic history, habits, and outcomes).
- `EdIneqFull.ipynb` / `index.html` – notebook and HTML export of earlier exploration.

Dataset at a glance
-------------------
Source: “Student Performance Metrics” (Mendeley Data: https://data.mendeley.com/datasets/5b82ytz489/1).  
Rows: 400 students (CSV header shown below).

Columns (selected):
- Demographics: `Gender`, `Hometown` (City/Village), `Income` (4 buckets in RM), `Department`.
- Academics before college: `HSC`, `SSC` (GPA on 5.0 scale), `English` proficiency, `Semester` currently enrolled.
- Habits: `Preparation`, `Gaming`, `Attendance`, `Job`, `Extra` (extracurriculars).
- Targets: `Overall` (college GPA on 4.0 scale), `Last` (most recent semester GPA).

What the analysis does
----------------------
1) **Data audit** – type checks, missing-value scan, summary statistics overall and by Income/Hometown/GPA bands.  
2) **Hypothesis tests**  
   - Chi‑square: time on gaming/job/extracurriculars vs. preparation time.  
   - One‑way ANOVA (+ Tukey HSD): Income level vs. Computer proficiency.  
   - Spearman correlation: High‑school GPA (`HSC`) vs. college GPA (`Overall`).  
3) **Visuals** – heatmaps, box plots, scatter + regression line.  
4) **Modeling** – baseline Linear Regression and a RandomForestRegressor (125 trees) to predict `Overall`. Metrics printed for train/test and plotted as predicted vs. actual GPA.

Environment setup
-----------------
Requires Python 3.9+ and the packages below.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas seaborn matplotlib scipy scikit-learn
```

How to run
----------
- **Full analysis & plots**: `python main.py`  
  (opens matplotlib windows; run in an environment with a display or use a backend like `Agg` if headless.)
- **Pipeline‑only regression**: `python test.py`  
  Prints MSE/R² and shows predicted vs. actual GPA scatter.
- **Teaching demos**: `python ml_example.py`  
  Generates synthetic datasets and compares regression fits.

Repro tips
----------
- Ensure `ResearchInformation3.csv` stays in the repo root; scripts read it relative to the working directory.  
- If running in a notebook, set `matplotlib` backend as needed and reuse code blocks from `main.py` or `EdIneqFull.ipynb`.  
- Random seeds are fixed (`random_state=42`) for reproducible splits and trees.

Ideas for future work
---------------------
- Add a `requirements.txt` and pin library versions for repeatable runs.  
- Log all metrics to a CSV/MLflow run for comparison across model variants.  
- Try regularized linear models, gradient boosting, and cross‑validation to reduce overfitting.  
- Add unit tests around preprocessing and metrics to guard against data/schema drift.
