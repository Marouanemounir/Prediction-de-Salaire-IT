# Validation locale — 28 octobre 2026

Base récupérée : `49e9886c8f7839c49ee25e7a032d68f101981604` du dépôt
`Marouanemounir/Prediction-de-Salaire-IT`. Environnement Windows, Python 3.12.10,
dépendances complètes dans `requirements-lock.txt`.

## Évaluation réellement exécutée

- Deux entraînements complets avec graine 2025, sorties `artifacts` puis
  `artifacts-check`.
- Identité octet par octet de `split_manifest.csv`, `model_comparison.csv`,
  `test_predictions.csv`, `metrics.json`, `cv_results.json`, `shap_global.csv`
  et `shap_local.csv` entre ces deux exécutions.
- Modèle choisi en CV : Lasso, alpha 1 ; test final de 998 lignes.
- R² 0,9246802635 ; MAE 1 142,946969 MAD/mois ; RMSE 1 469,885934 MAD/mois.
- Rayon conformal 2 433,230982 MAD/mois ; couverture test 0,9198396794.
- Pas de garantie à 90 % dans chaque sous-groupe : par exemple, couverture
  observée 83,17 % pour les 101 lignes « Ingénieur Cybersécurité ».

## Vérifications techniques

`python -m pytest -q` : **11 tests réussis** avec les artefacts locaux présents.
Les contrôles couvrent les partitions/plis, le prétraitement, la régression
NumPy, le quantile conformal, le schéma, SHAP, les profils hors domaine,
l'intégrité du modèle et la reproduction des prédictions sauvegardées.

Le test Streamlit utilise `streamlit.testing.v1.AppTest` : chargement des trois
onglets, soumission du formulaire, affichage de l'estimation et de SHAP sans
exception. Il s'agit d'un test de l'application, pas d'un test visuel dans un
navigateur. Les deux graphiques de diagnostic ont été inspectés visuellement.

Des avertissements de dépréciation pyparsing/Matplotlib sont émis par les
dépendances ; ils ne font pas échouer les tests. Les tests d'intégration sont
ignorés sur un clone sans modèle local ; les tests unitaires restent exécutables.

Au moment de cette validation locale initiale, le workflow GitHub Actions
n'avait pas encore été lancé sur GitHub. Son statut après publication est
consultable dans l'onglet Actions du dépôt.
La provenance du CSV, sa représentativité et les performances externes restent
non établies.
