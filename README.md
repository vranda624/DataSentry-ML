DataSentry ML
CPU-only data-quality profiling, corruption detection, and model evaluation dashboard.

DataSentry ML helps inspect tabular datasets, surface common data-quality issues, and compare baseline machine-learning models for identifying potentially corrupted records. It runs locally on Windows and does not require a GPU or cloud service.
Important: The included demo dataset and labels are synthetic. Metrics measured on this dataset demonstrate the pipeline on this specific synthetic example; they do not establish real-world accuracy.

Features
- Upload and inspect CSV datasets in a Streamlit dashboard.
- Profile dataset completeness and identify common quality issues.
- Review data-quality findings and explore records.
- Run CPU-friendly baseline classifiers:
  - Logistic Regression
  - Random Forest
- Evaluate models with accuracy, precision, recall, F1-score, confusion matrices, and classification reports.
- Tune a classification threshold using a validation split.
- Export evaluation reports in CSV, TXT, and JSON formats.
- Run unit tests for the core profiling and corruption utilities.
Technology stack
- Language: Python
- Data processing: Pandas, NumPy
- Machine learning: Scikit-learn
- Dashboard: Streamlit
- Visualization: Plotly
- Testing: Pytest
- Data files: CSV; OpenPyXL is included for Excel support where used
Project structure
datasentry_ml/
├── app.py
├── datasentry/
│   ├── __init__.py
│   ├── profiler.py
│   ├── corruption.py
│   ├── ml_model.py
│   └── best_model.joblib        # generated after training
├── tests/
│   ├── test_profiler.py
│   └── test_corruption.py
├── reports/                     # generated after training
│   ├── model_metrics.csv
│   ├── model_evaluation_report.txt
│   └── model_evaluation_report.json
├── datasentry_demo_20000.csv    # optional demo dataset
├── datasentry_demo_20000_labels.csv
├── datasentry_demo_20000_corruption_notes.csv
├── requirements.txt
├── README.md
└── .gitignore
Some generated files and optional demo files may not exist until you create or run them.
Requirements
- Windows 10/11 or another OS supported by Python
- Python 3.11+ recommended (use a Python version supported by the installed Scikit-learn release)
- 4 GB RAM can run the demo pipeline; other datasets may need more memory
- VS Code is recommended but not required
Setup on Windows (VS Code)
Open a terminal in the project root—the folder containing app.py and requirements.txt.
1. Activate the virtual environment
If your existing terminal already starts with (.venv), skip this step.

For Command Prompt:
..\.venv\Scripts\activate
If the virtual environment is not present, create it from the project root:
python -m venv .venv
.venv\Scripts\activate
2. Install dependencies
Use Python to invoke pip:
python -m pip install -r requirements.txt
If your computer is managed by an organization and its security policy blocks package installation, do not disable that policy. Ask the device administrator or use an approved Python environment.
3. Run the dashboard
python -m streamlit run app.py
Open the local URL printed by Streamlit, usually:
http://localhost:8501
Stop the dashboard with Ctrl+C in the terminal.
Train and evaluate the CPU-only models
Place these files in the project root if you want to use the supplied demo dataset:
- datasentry_demo_20000.csv
- datasentry_demo_20000_labels.csv
The corruption notes file is useful for auditing the synthetic injections but is not required by the training script.

From the project root, run:
python -m datasentry.ml_model
The training script compares Logistic Regression and Random Forest on a held-out split, selects a threshold using validation data, and writes evaluation reports to reports/. It also saves a model artifact at datasentry/best_model.joblib.

If the module cannot be found, verify that ml_model.py is saved inside the datasentry folder and that the terminal is in the project root. If your current ml_model.py is in the project root instead, either move it into datasentry/ or run it using its actual path and adapt imports accordingly.
Generated evaluation files
File	Contents
reports/model_metrics.csv	Compact comparison of model metrics
reports/model_evaluation_report.txt	Human-readable report with confusion matrices and class metrics
reports/model_evaluation_report.json	Structured metrics for future integrations
datasentry/best_model.joblib	Saved model artifact produced by training


These are generated outputs; they may be absent before the training command succeeds.
Run tests
From the project root:
python -m pytest -q
The original starter tests should report four passing tests when the project and dependencies are correctly installed.
Understanding the metrics
- Accuracy: fraction of all records classified correctly.
- Precision: fraction of records predicted as corrupted that are actually labelled corrupted.
- Recall: fraction of labelled corrupted records the model detects.
- F1-score: harmonic mean of precision and recall.
- Confusion matrix: counts true negatives, false positives, false negatives, and true positives.
For data-quality monitoring, recall is important because missed corruptions may remain undetected; precision is also important because false alarms create review work. Choose metrics based on the cost of each type of error rather than maximizing accuracy alone.
Data and evaluation limitations
- The demo dataset and its labels are synthetic and contain deliberately injected issues.
- Some engineered features directly encode known quality rules (for example, missing values or invalid ranges). When those rules also define the labels, high scores may reflect learning the same rules rather than discovering unknown errors.
- A row-level target is used by the current baseline: a record is labelled positive if any cell in its corresponding label row is flagged.
- Threshold tuning must use validation data, not the test set.
- For an unbiased final estimate, choose the model using validation results and evaluate it once on a separate, untouched test set. Do not use test-set scores to select the final model and then present those same scores as an unbiased final result.
- Results on synthetic data do not guarantee performance on real-world datasets. Validate against representative, independently labelled data before making operational claims.
- The current baseline is not a replacement for deterministic data-validation rules, data governance, or human review.
Troubleshooting
No module named datasentry.ml_model
Check that ml_model.py is saved inside datasentry/, that datasentry/__init__.py exists, and that the terminal is at the project root. Then retry:
python -m datasentry.ml_model
No module named streamlit or another dependency
Run:
python -m pip install -r requirements.txt
FileNotFoundError for the demo dataset
Confirm the CSV files are in the same folder as app.py, then check:
dir *.csv
Port 8501 is already in use
Stop the other Streamlit process with Ctrl+C, or launch on another port:
python -m streamlit run app.py --server.port 8502
Suggested future improvements
- Validate rules and model predictions separately.
- Add schema-aware checks and duplicate detection across batches.
- Add drift checks between a reference dataset and a new batch.
- Add explainable, record-level findings and exportable remediation suggestions.
- Evaluate on independently labelled, real-world datasets.
- Add validation-only model selection and preserve an untouched final test set.
License
Add a license before publishing or distributing this project. Until a license is included, do not assume others have permission to reuse the code.