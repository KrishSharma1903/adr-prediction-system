import json
from pathlib import Path

import joblib
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

SEX_OPTIONS = {"Female": "F", "Male": "M", "Not reported": "Unknown"}
LOW_DRUG_DATA = 1000
LOW_REACTION_DATA = 500
MAX_REASONS = 5

bundle = joblib.load(MODELS_DIR / "serious_best_model.joblib")
config = json.loads((MODELS_DIR / "serious_feature_config.json").read_text())

model = bundle["model"]
model_name = bundle["model_name"]
threshold = bundle["threshold"]
test_metrics = bundle["test_metrics"]
all_metrics = pd.read_csv(MODELS_DIR / "serious_metrics.csv", index_col=0)
features = config["feature_columns"]

drug_names = [n.replace("\\", " + ").title() for n in config["ingredients"]]
drug_to_column = dict(zip(drug_names, config["ingredient_columns"]))
drug_reports = dict(zip(drug_names, config["ingredient_report_counts"]))
reaction_to_column = dict(zip(config["reactions"], config["reaction_columns"]))
reaction_reports = dict(zip(config["reactions"], config["reaction_report_counts"]))

labels = {
    "age_years": "Age",
    "age_missing": "Age not reported",
    "sex_F": "Female",
    "sex_M": "Male",
    "sex_Unknown": "Sex not reported",
    "n_drugs": "Number of drugs",
    "n_other_drugs": "Unlisted drugs",
    "n_reactions": "Number of reactions",
    "n_other_reactions": "Unlisted reactions",
}

for name, col in drug_to_column.items():
    labels[col] = name

for name, col in reaction_to_column.items():
    labels[col] = f"Reaction: {name}"

explainer = shap.TreeExplainer(model)
