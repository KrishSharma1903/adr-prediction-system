from html import escape

import numpy as np
import pandas as pd
import streamlit as st

import serious_predictor as sp

SERIOUS_COLOR = "orangered"
NON_SERIOUS_COLOR = "royalblue"

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1200px; }}
h1 {{ font-size: 2.1rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin-bottom: 0 !important; }}
.subtitle {{ opacity: 0.7; font-size: 1.02rem; margin: 0.3rem 0 1rem 0; }}
.note {{ border-left: 3px solid rgba(128,128,128,0.45); padding: 0.55rem 0.9rem; font-size: 0.86rem; opacity: 0.8;
         background: rgba(128,128,128,0.06); border-radius: 0 6px 6px 0; margin-bottom: 1.2rem; }}
.eyebrow {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; font-weight: 600; opacity: 0.6; margin-bottom: 0.35rem; }}
.card {{ border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }}
.verdict-row {{ display: flex; align-items: baseline; gap: 0.9rem; flex-wrap: wrap; }}
.pill {{ color: white; font-weight: 700; font-size: 1.5rem; padding: 0.15rem 0.9rem; border-radius: 999px; }}
.prob {{ font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; }}
.prob-label {{ opacity: 0.65; font-size: 0.92rem; }}
.meter {{ position: relative; height: 10px; border-radius: 999px; background: rgba(128,128,128,0.18); margin: 1rem 0 0.35rem 0; }}
.meter-fill {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }}
.meter-mark {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: currentColor; opacity: 0.55; }}
.meter-scale {{ display: flex; justify-content: space-between; font-size: 0.75rem; opacity: 0.55; }}
.meta {{ font-size: 0.82rem; opacity: 0.6; margin-top: 0.8rem; }}
.reason {{ display: grid; grid-template-columns: 6.2rem 1fr 7rem; align-items: center; gap: 0.8rem;
           padding: 0.55rem 0; border-bottom: 1px solid rgba(128,128,128,0.15); }}
.reason:last-child {{ border-bottom: none; }}
.tag {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; text-align: center;
        padding: 0.18rem 0; border-radius: 6px; }}
.tag-up {{ color: {SERIOUS_COLOR}; background: color-mix(in srgb, {SERIOUS_COLOR} 12%, transparent); }}
.tag-down {{ color: {NON_SERIOUS_COLOR}; background: color-mix(in srgb, {NON_SERIOUS_COLOR} 12%, transparent); }}
.factor {{ font-weight: 500; }}
.factor small {{ display: block; font-weight: 400; opacity: 0.6; }}
.strength {{ height: 6px; border-radius: 999px; background: rgba(128,128,128,0.15); overflow: hidden; }}
.strength div {{ height: 100%; border-radius: 999px; }}
.empty {{ border: 1px dashed rgba(128,128,128,0.4); border-radius: 12px; padding: 2.2rem 1.6rem; text-align: center; opacity: 0.75; }}
.step {{ border: 1px solid rgba(128,128,128,0.25); border-radius: 10px; padding: 0.9rem 1rem; height: 100%; }}
.step b {{ display: block; margin-bottom: 0.3rem; }}
.step span {{ font-size: 0.86rem; opacity: 0.7; }}
</style>
"""

st.set_page_config(page_title="ADR Seriousness Predictor", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


drug_names = sorted(sp.drug_names)
reaction_names = sorted(sp.reaction_to_column)
metrics = sp.test_metrics

defaults = {
    "age": 50,
    "age_unknown": False,
    "sex": "Female",
    "drugs": [],
    "other_drugs": 0,
    "total_drugs": 1,
    "reactions": [],
    "other_reactions": 0,
}

for key, value in defaults.items():
    st.session_state.setdefault(key, value)

st.title("Adverse Event Seriousness Predictor")

st.markdown(
    '<div class="subtitle">Estimates whether a reported adverse drug event is likely to be serious '
    "(death, hospitalisation, life-threatening, disability), and explains which details drove the prediction.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="note"><b>Not medical advice.</b> Based on patterns in reports to the FDA Adverse Event Reporting System (FAERS). '
    "A prediction shows how similar reports were recorded; it does not show that a drug caused a reaction and must not be used "
    "for diagnosis or treatment decisions.</div>",
    unsafe_allow_html=True,
)

input_col, result_col = st.columns([1, 1.25], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Patient</div>', unsafe_allow_html=True)
        patient_cols = st.columns(2)

        age = patient_cols[0].number_input(
            "Age (years)",
            min_value=0,
            max_value=120,
            step=1,
            key="age",
            disabled=st.session_state.age_unknown,
        )
        sex = patient_cols[1].selectbox("Sex", list(sp.SEX_OPTIONS), key="sex")
        age_unknown = st.checkbox("Age not known", key="age_unknown")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Drugs</div>', unsafe_allow_html=True)

        drugs = st.multiselect(
            "Main drugs (up to 3)",
            drug_names,
            max_selections=3,
            key="drugs",
            placeholder="Search active ingredients",
            help="The 500 most commonly reported active ingredients. Put the main suspect drug first.",
        )

        drug_cols = st.columns(2)
        other_drugs = drug_cols[0].number_input("Main drugs not listed", 0, 3, step=1, key="other_drugs")
        total_drugs = drug_cols[1].number_input("Total drugs taken", 1, 300, step=1, key="total_drugs")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Reactions</div>', unsafe_allow_html=True)

        reactions = st.multiselect(
            "Reported reactions",
            reaction_names,
            key="reactions",
            placeholder="Search reactions",
            help="The 300 most commonly reported reactions. Outcome terms such as 'Death' are not inputs.",
        )

        other_reactions = st.number_input("Reactions not listed", 0, 200, step=1, key="other_reactions")

    other_drugs = min(other_drugs, 3 - len(drugs))
    total_drugs = max(total_drugs, len(drugs) + other_drugs, 1)

with result_col:
    has_drug = bool(drugs or other_drugs)
    has_reaction = bool(reactions or other_reactions)

    if not has_drug or not has_reaction:
        st.markdown(
            '<div class="empty"><b>No prediction yet</b><br>Add at least one drug and one reaction '
            "to see a prediction.</div>",
            unsafe_allow_html=True,
        )
    else:
        row = dict.fromkeys(sp.features, 0.0)

        row["age_years"] = np.nan if age_unknown else float(age)
        row["age_missing"] = float(age_unknown)
        row[f"sex_{sp.SEX_OPTIONS[sex]}"] = 1.0

        row["n_drugs"] = float(total_drugs)
        row["n_other_drugs"] = float(other_drugs)
        row["n_reactions"] = float(len(reactions) + other_reactions)
        row["n_other_reactions"] = float(other_reactions)

        for drug in drugs:
            row[sp.drug_to_column[drug]] = 1.0
        for reaction in reactions:
            row[sp.reaction_to_column[reaction]] = 1.0

        feature_table = pd.DataFrame([row], columns=sp.features)

        probability = round(float(sp.model.predict_proba(feature_table)[0, 1]), 4)
        prediction = "Serious" if probability >= sp.threshold else "Non-serious"

        shap_values = pd.Series(sp.explainer.shap_values(feature_table)[0], index=sp.features)
        event = feature_table.iloc[0]

        describing = ["n_drugs", "n_reactions"]
        if not pd.isna(event["age_years"]):
            describing.append("age_years")
        for count in ["n_other_drugs", "n_other_reactions"]:
            if event[count] > 0:
                describing.append(count)

        numeric = ("age_years", "n_drugs", "n_reactions", "n_other_drugs", "n_other_reactions")
        describing += [f for f in sp.features if f not in numeric and event[f] == 1]

        relevant = shap_values[describing]
        relevant = relevant[relevant.abs() >= 0.02]
        biggest_first = relevant.abs().sort_values(ascending=False).index
        relevant = relevant.reindex(biggest_first).head(sp.MAX_REASONS)

        color = SERIOUS_COLOR if prediction == "Serious" else NON_SERIOUS_COLOR
        threshold = sp.threshold * 100

        verdict_html = f"""
        <div class="card">
          <div class="eyebrow">Prediction</div>
          <div class="verdict-row">
            <span class="pill" style="background:{color}">{escape(prediction)}</span>
            <span class="prob">{probability * 100:.1f}%</span>
            <span class="prob-label">probability of a serious outcome</span>
          </div>
          <div class="meter">
            <div class="meter-fill" style="width:{probability * 100:.1f}%; background:{color}"></div>
            <div class="meter-mark" style="left:{threshold:.0f}%"></div>
          </div>
          <div class="meter-scale"><span>0%</span><span>threshold {threshold:.0f}%</span><span>100%</span></div>
          <div class="meta">{escape(sp.model_name)} &middot; test accuracy {metrics['accuracy']:.0%} &middot;
            recall {metrics['recall']:.0%} &middot; ROC-AUC {metrics['roc_auc']:.2f}</div>
        </div>
        """

        verdict_html = "".join(line.strip() for line in verdict_html.splitlines())
        st.markdown(verdict_html, unsafe_allow_html=True)

        for drug in drugs:
            if sp.drug_reports[drug] < sp.LOW_DRUG_DATA:
                st.warning(
                    f"{drug} appears in only {sp.drug_reports[drug]:,} training reports, "
                    "so this prediction is less certain."
                )

        for reaction in reactions:
            if sp.reaction_reports[reaction] < sp.LOW_REACTION_DATA:
                st.warning(
                    f"Reaction '{reaction}' appears in only {sp.reaction_reports[reaction]:,} training reports, "
                    "so this prediction is less certain."
                )

        if len(relevant) > 0:
            largest = max(abs(round(float(v), 4)) for v in relevant)
            rows_html = ""

            for feature, value in relevant.items():
                value = round(float(value), 4)
                raw = event[feature]

                if feature == "age_years":
                    factor = f"Age {raw:.0f}"
                elif feature == "n_drugs":
                    factor = f"Taking {raw:.0f} drug" + ("" if raw == 1 else "s")
                elif feature == "n_reactions":
                    factor = f"{raw:.0f} reaction" + ("" if raw == 1 else "s") + " reported"
                elif feature == "n_other_drugs":
                    factor = f"{raw:.0f} unlisted drug(s)"
                elif feature == "n_other_reactions":
                    factor = f"{raw:.0f} unlisted reaction(s)"
                else:
                    factor = sp.labels[feature]

                up = value > 0
                size = abs(value)

                if size >= 0.5:
                    strength = "strong effect"
                elif size >= 0.15:
                    strength = "moderate effect"
                else:
                    strength = "slight effect"

                bar_color = SERIOUS_COLOR if up else NON_SERIOUS_COLOR
                tag_class = "tag-up" if up else "tag-down"
                tag_text = "Raises" if up else "Lowers"
                bar_width = size / largest * 100

                rows_html += f"""
                <div class="reason">
                  <span class="tag {tag_class}">{tag_text}</span>
                  <span class="factor">{escape(factor)}<small>{strength}</small></span>
                  <span class="strength"><div style="width:{bar_width:.0f}%; background:{bar_color}"></div></span>
                </div>"""

            reasons_html = f'<div class="card"><div class="eyebrow">Why this prediction</div>{rows_html}</div>'
            reasons_html = "".join(line.strip() for line in reasons_html.splitlines())
            st.markdown(reasons_html, unsafe_allow_html=True)
            st.caption("Bars show the relative size of each detail's effect, from SHAP values.")
        else:
            st.info("No single detail stood out; the prediction is close to the average report.")

        if drugs:
            st.caption("Training reports: " + " · ".join(f"{d} {sp.drug_reports[d]:,}" for d in drugs))

st.divider()
st.subheader("How it works")

steps = [
    ("1. Data", "FAERS 2025 Q1: demographics, drugs, reactions and outcomes, cleaned and de-duplicated."),
    ("2. Target", "A report is serious if it has an outcome: death, hospitalisation, life-threatening, disability, and others."),
    ("3. Models", "Random Forest, XGBoost and SVM compared on the same data; the best on validation F1 is used."),
    ("4. Explanation", "SHAP values show how each detail of the event pushed the prediction up or down."),
]

for col, (title, text) in zip(st.columns(4), steps):
    col.markdown(f'<div class="step"><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)

st.subheader("Model comparison")

comparison = sp.all_metrics.drop(columns=["train_seconds"], errors="ignore")
comparison.columns = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

st.caption(
    f"Scores on 77,356 test reports the models never saw. Selected model: {sp.model_name} "
    "(highest F1 on the validation set). Recall = share of truly serious reports that were caught."
)
