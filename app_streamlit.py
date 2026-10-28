"""Interactive reader of the reproducible salary study and SHAP predictions."""
from pathlib import Path
import pandas as pd
import streamlit as st
from salary_study.data import CATEGORICAL
from salary_study.explain import explain_predictions
from salary_study.inference import load_bundle, predict

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Étude de rémunération IT", page_icon="💼", layout="wide")
st.title("Étude exploratoire de rémunération IT — Maroc")
st.warning("Source des données non vérifiée : ce prototype ne constitue pas une enquête représentative du marché. "
           "Unité annoncée : MAD/mois ; brut/net et primes non documentés.")


@st.cache_resource
def cached_bundle(model_mtime, metadata_mtime):
    return load_bundle(ARTIFACTS)


try:
    bundle, metadata = cached_bundle((ARTIFACTS / "model.joblib").stat().st_mtime_ns,
                                     (ARTIFACTS / "metadata.json").stat().st_mtime_ns)
except FileNotFoundError:
    st.info("Entraîner d'abord les modèles depuis le dossier du projet :")
    st.code("python -m salary_study.train", language="bash")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

study, simulation, diagnostics = st.tabs(["Résultats de l'étude", "Simulation et SHAP", "Diagnostics et données"])
with study:
    st.subheader(f"Modèle retenu : {metadata['selected_model']}")
    st.caption("Sélection sur la MAE en validation croisée du seul entraînement. Le test reste distinct.")
    score = metadata["selected_test_metrics"]
    columns = st.columns(3)
    columns[0].metric("R² test", f"{score['r2']:.4f}")
    columns[1].metric("MAE test (MAD/mois)", f"{score['mae_mad']:,.0f}")
    columns[2].metric("Couverture test de l'intervalle", f"{metadata['interval']['test_coverage']:.1%}")
    st.caption("Le R² mesure la variance expliquée, pas un pourcentage de prédictions exactes.")
    st.dataframe(pd.read_csv(ARTIFACTS / "model_comparison.csv"), hide_index=True, width="stretch")
    st.markdown("**Médianes et quartiles du fichier** — repères descriptifs sans pondération du marché.")
    st.dataframe(pd.read_csv(ARTIFACTS / "salary_segments.csv"), hide_index=True, width="stretch")
    st.download_button("Télécharger le rapport client", (ARTIFACTS / "rapport_client.md").read_text(encoding="utf-8"),
                       file_name="rapport_remuneration_IT.md", mime="text/markdown")

with simulation:
    st.subheader("Simuler un profil présent dans le domaine d'entraînement")
    labels = {"Entreprise": "Entreprise", "Profil": "Profil IT", "Niveau_Etude": "Niveau d'étude", "Technologie": "Technologie"}
    with st.form("prediction"):
        inputs = {c: st.selectbox(labels[c], bundle["categories"][c]) for c in CATEGORICAL}
        inputs["Experience"] = st.slider("Années d'expérience", float(bundle["experience_range"][0]),
                                         float(bundle["experience_range"][1]), 5.0, 0.5)
        submitted = st.form_submit_button("Estimer et expliquer", type="primary")
    if submitted:
        rows = pd.DataFrame([inputs])
        try:
            point, lower, upper = predict(bundle, rows)
            st.metric("Estimation (MAD/mois)", f"{point[0]:,.0f}")
            if point[0] <= 0:
                st.warning("Estimation non positive : ce profil révèle une limite du modèle. "
                           "Cette estimation ne peut pas servir de proposition salariale.")
            st.info(f"Intervalle de prédiction nominal à {1-bundle['alpha']:.0%} : "
                    f"{lower[0]:,.0f} à {upper[0]:,.0f} MAD/mois.")
            st.caption("Couverture marginale sous échangeabilité, sans garantie par profil ou sur le marché réel. "
                       "Largeur constante, calibrée sur un jeu distinct. Les bornes ne sont pas tronquées.")
            if (inputs["Profil"], inputs["Technologie"]) not in bundle["profile_technology_pairs"]:
                st.warning("Cette combinaison profil/technologie est absente de l'entraînement. "
                           "La simulation repose sur une combinaison non observée.")
            contributions, base = explain_predictions(bundle["pipeline"], bundle["background"], rows)
            st.subheader("Contributions SHAP à cette estimation")
            st.caption(f"Référence du modèle : {base[0]:,.0f} MAD/mois. "
                       "Référence + somme des contributions = estimation. Associations du modèle, sans causalité.")
            st.bar_chart(contributions.T.rename(columns={0: "Contribution (MAD/mois)"}), horizontal=True)
            st.dataframe(contributions.T.rename(columns={0: "Contribution (MAD/mois)"}), width="stretch")
        except ValueError as error:
            st.error(str(error))

with diagnostics:
    st.image(str(ARTIFACTS / "residual_diagnostics.png"))
    st.image(str(ARTIFACTS / "shap_importance.png"))
    st.json(metadata["residual_diagnostics"])
    st.dataframe(pd.read_csv(ARTIFACTS / "segments_test.csv"), hide_index=True, width="stretch")
    st.caption("Petits sous-groupes : résultats instables. Tests exploratoires, p-values non corrigées.")
    st.json({"audit": metadata["data_audit"], "tailles": metadata["split_sizes"],
             "IC bootstrap 95 % des métriques": metadata["selected_test_bootstrap_95"]})
    st.markdown((ROOT / "docs" / "DATA_CARD.md").read_text(encoding="utf-8"))
