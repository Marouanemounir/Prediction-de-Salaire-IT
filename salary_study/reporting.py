"""Client-facing exploratory report and static diagnostic figures."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def figures(directory, predictions, importance):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    y, pred, residual = (predictions[c] for c in ["actual_mad", "prediction_mad", "residual_mad"])
    axes[0, 0].scatter(y, pred, s=9, alpha=.35)
    limits = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    axes[0, 0].plot(limits, limits, "k--")
    axes[0, 0].set(xlabel="Salaire observé (MAD/mois)", ylabel="Prédiction (MAD/mois)", title="Test : observé / prédit")
    axes[0, 1].scatter(pred, residual, s=9, alpha=.35)
    axes[0, 1].axhline(0, color="black", linestyle="--")
    axes[0, 1].set(xlabel="Prédiction (MAD/mois)", ylabel="Observé - prédit (MAD/mois)", title="Résidus sur test")
    axes[1, 0].hist(residual, bins=35, edgecolor="white")
    axes[1, 0].set(xlabel="Résidu (MAD/mois)", ylabel="Effectif", title="Distribution des résidus")
    stats.probplot(residual, dist="norm", plot=axes[1, 1])
    axes[1, 1].set(title="Q-Q plot des résidus", xlabel="Quantile normal théorique", ylabel="Résidu ordonné")
    fig.savefig(directory / "residual_diagnostics.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.barh(importance.variable, importance.mean_abs_shap_mad, color="#147d92")
    ax.set(xlabel="Moyenne |SHAP agrégé| (MAD/mois)", title="Contributions du modèle retenu (échantillon test)")
    fig.savefig(directory / "shap_importance.png", dpi=160)
    plt.close(fig)


def client_report(directory, metadata, comparison, segments):
    best = metadata["selected_model"]
    score = metadata["selected_test_metrics"]
    interval = metadata["interval"]
    audit = metadata["data_audit"]
    rows = "\n".join(f"| {r.model} | {r.cv_mae_mad:,.0f} | {r.test_r2:.4f} | {r.test_mae_mad:,.0f} | {r.test_rmse_mad:,.0f} |"
                     for r in comparison.itertuples())
    segment_rows = "\n".join(f"| {r.Profil} | {r.n} | {r.median_mad:,.0f} | {r.q25_mad:,.0f}–{r.q75_mad:,.0f} |"
                             for r in segments.itertuples())
    tests = metadata["residual_diagnostics"]
    test_lines = "\n".join(f"- {name} : statistique {tests[name]['statistic']:.3f}, p = {tests[name]['p_value']:.3g}. H0 : {tests[name]['null']}. "
                            f"{'Rejet de H0' if tests[name]['p_value'] < .05 else 'H0 non rejetée'} au seuil exploratoire de 5 %."
                            for name in ["jarque_bera", "bias_t_test", "breusch_pagan_koenker"])
    ci = metadata["selected_test_bootstrap_95"]
    report = f"""# Étude exploratoire de rémunération IT — Maroc

## Note au client

Ce livrable évalue un prototype de simulation sur le CSV fourni dans le dépôt.
**La provenance des observations n'est pas vérifiée. Ce document ne constitue pas
une enquête représentative du marché marocain, ni une grille salariale validée.**
L'unité MAD/mois vient du dépôt ; brut/net, primes, date réelle de collecte et
type de contrat ne sont pas documentés. Le nom du fichier mentionne 2025 sans
preuve de collecte correspondante.

## Résultats et usage

Le modèle **{best}** est retenu sur la MAE moyenne en validation croisée à 5 plis
du jeu d'entraînement uniquement. Test final : R² **{score['r2']:.4f}**,
MAE **{score['mae_mad']:,.0f} MAD/mois**, RMSE **{score['rmse_mad']:,.0f} MAD/mois**.
Le R² mesure la variance expliquée ; il ne mesure pas un pourcentage de salaires exacts.
Bootstrap conditionnel au modèle entraîné (1 000 réplications), IC à 95 % :
R² [{ci['r2'][0]:.4f} ; {ci['r2'][1]:.4f}],
MAE [{ci['mae_mad'][0]:,.0f} ; {ci['mae_mad'][1]:,.0f}] MAD/mois.
Ces intervalles ne mesurent pas l'incertitude liée au réentraînement ni aux biais de collecte.

| Modèle | MAE CV (MAD) | R² test | MAE test (MAD) | RMSE test (MAD) |
|---|---:|---:|---:|---:|
{rows}

Les scores test des autres modèles sont fournis pour audit ; ils ne servent ni
à choisir le gagnant ni à modifier la grille. Les grilles et scores par pli sont
dans `cv_results.json`. La baseline prédit la médiane d'entraînement.
Les intervalles de confiance des métriques des autres modèles sont dans `metrics.json`.
Des scores proches ne démontrent pas une supériorité statistique entre modèles.

## Fourchettes individuelles

Intervalle conformal nominal à {(1-interval['alpha'])*100:.0f} % : prédiction ±
**{interval['radius_mad']:,.0f} MAD/mois**. Couverture observée sur le test :
**{interval['test_coverage']*100:.1f} %**, largeur moyenne **{interval['mean_width_mad']:,.0f} MAD/mois**.
Il s'agit d'un intervalle de prédiction individuel, distinct de l'IC d'une métrique.
La largeur est constante : elle peut mal refléter certains profils.
La garantie conformale standard est marginale et suppose des observations
échangeables. Le regroupement des profils, les dépendances entre salariés et le
passage au marché réel limitent cette garantie ; seule la couverture empirique
ci-dessus est mesurée ici. Les bornes ne sont pas tronquées à zéro.

La couverture peut être plus faible dans certains sous-groupes : consulter
`segments_test.csv` avant d'utiliser une fourchette individuelle.

## Diagnostic des erreurs

Résidu = salaire observé - salaire prédit. Moyenne :
{tests['mean_residual_mad']:,.0f} MAD/mois. Tests exploratoires sur le test final :
{test_lines}

Seuil descriptif de lecture : 0,05 ; p-values non corrigées pour tests multiples.
Un rejet de normalité n'interdit pas la prédiction ; l'hétéroscédasticité appelle
une lecture prudente de la largeur constante. Sans identifiants ou dates,
l'indépendance des observations n'est pas vérifiable. Les tests ne prouvent pas
la représentativité. Voir `residual_diagnostics.png`, `diagnostics.json` et
`segments_test.csv` (erreur et couverture par profil et niveau d'expérience).

![Diagnostic des résidus sur le test](residual_diagnostics.png)

## Repères descriptifs du fichier

{audit['raw_rows']} lignes brutes ; {audit['exact_duplicates_removed']} doublons exacts exclus ;
{audit['usable_rows']} lignes exploitées. Salaire observé :
{audit['salary_min']:,.0f}–{audit['salary_max']:,.0f} MAD/mois ;
médiane {audit['salary_median']:,.0f} MAD/mois.
Les repères ci-dessous décrivent toutes les lignes dédupliquées du fichier,
sans pondération, après gel du modèle ; ils ne sont pas des estimations du marché.
L'intervalle Q25–Q75 est descriptif, pas un intervalle de prédiction.

| Profil | Effectif | Médiane MAD/mois | Q25–Q75 MAD/mois |
|---|---:|---:|---:|
{segment_rows}

## Explication et décision

Les contributions SHAP sont calculées sur un échantillon déterministe de
{metadata['shap_sample_size']} lignes test, avec 100 lignes d'entraînement comme référence.
Les contributions des colonnes encodées sont additionnées par variable source.
Référence + somme des contributions = prédiction, contrôlé numériquement.
Voir `shap_importance.png`, `shap_global.csv` et `shap_local.csv`.
Les dépendances profil/technologie compliquent l'attribution : SHAP explique
le modèle sous une convention interventionnelle, sans établir de causalité.

![Importance SHAP des cinq variables](shap_importance.png)

Pour une utilisation RH, commencer par une enquête documentée couvrant PME,
ESN, entreprises locales, villes, contrats et ancienneté, avec base brut/net et
primes harmonisées. Valider ensuite sur une collecte externe récente et sur
des entreprises absentes de l'apprentissage. Ne pas décider d'une rémunération
individuelle à partir de ce prototype.

## Traçabilité

Graine : {metadata['seed']} ; tailles : {metadata['split_sizes']}.
Séparation d'environ 60/20/20 par groupes de variables d'entrée identiques.
Les prétraitements sont appris dans chaque pli d'entraînement, puis sur le
jeu d'entraînement final. Ni calibration ni test ne sont utilisés pour ajuster
les modèles ; calibration utilisée uniquement pour le rayon conformal.
Pas de réentraînement après calibration. Indices CSV (base 0) : `split_manifest.csv`.
SHA-256 des données : `{audit['sha256']}`.
Paramètres, versions, empreinte du modèle : `metadata.json`.
Provenance et limites détaillées : `../docs/DATA_CARD.md`.
"""
    (directory / "rapport_client.md").write_text(report, encoding="utf-8")
