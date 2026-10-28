# Explication du code et des changements

## Fonctionnement du projet d'origine

Le notebook lisait le CSV avec Pandas, créait toutes les colonnes one-hot,
séparait 80 % pour l'entraînement et 20 % pour le test, puis calculait moyenne
et écart-type sur l'entraînement. Une colonne de 1 servait d'intercept.

La prédiction était `X @ theta`. La fonction de coût valait
`sum((prediction-y)**2)/(2*n)`. Son gradient était
`X.T @ (prediction-y)/n`. À chaque itération, le programme effectuait
`theta = theta - learning_rate * gradient`. Les paramètres étaient sauvegardés
dans un pickle. Streamlit reconstruisait manuellement les colonnes et appliquait
la même formule à un profil saisi.

Ce socle était pertinent pour apprendre la régression. Les limites relevées
étaient l'encodage appris avant séparation, une initialisation aléatoire non
fixée, l'absence de comparaison et d'intervalles, la répétition de logique
d'inférence et l'absence de traçabilité des données. Normaliser avec la moyenne
du train était déjà une bonne pratique. Le jeu de test existait déjà ; l'objectif
est désormais de rendre sa construction et son exploitation auditables.

## Le nouveau parcours du code

### 1. `salary_study/data.py` : données et séparation

`load_data` contrôle les colonnes, les catégories vides, les valeurs manquantes,
les nombres non finis, l'expérience et les salaires. Il calcule l'empreinte
SHA-256 et un audit, puis enlève les doublons exacts. Le CSV reste intact.

`feature_groups` associe un groupe à chaque combinaison des cinq variables
d'entrée. `split_data` conserve chaque groupe dans une seule partition :
entraînement, calibration ou test. Cela empêche un profil d'entrée identique
d'apparaître de part et d'autre de l'évaluation. Ce n'est pas une séparation par
entreprise : prédire une entreprise jamais observée nécessiterait une autre étude.

### 2. `salary_study/models.py` : prétraitements et sept références

`make_pipeline` assemble un `ColumnTransformer` et le régresseur. L'expérience
est centrée et réduite ; les quatre variables catégorielles sont encodées en
one-hot avec une modalité de référence supprimée. Le Pipeline apprend ces
transformations à l'intérieur de chaque pli de validation.

`candidates` définit les recherches avant de voir le test :

| Modèle | Rôle / paramètres comparés |
|---|---|
| Médiane | Référence minimale pour vérifier l'utilité des variables |
| Linear Regression | Régression linéaire de scikit-learn |
| Linéaire NumPy | Descente de gradient pédagogique |
| Ridge | Pénalisation L2 ; alpha 0,1 / 10 / 100 |
| Lasso | Pénalisation L1 ; alpha 1 / 10 / 100 |
| Random Forest | 200 arbres ; feuilles min 1 / 5 ; variables 100 % / 70 % |
| Gradient Boosting | 150 / 300 arbres ; profondeur 2 / 3 ; taux 0,05 |

`ScratchLinearRegression` reste une vraie optimisation par gradient NumPy.
Il met en cache `X.T@X/n` et `X.T@y/n`, ce qui évite de parcourir toutes les
lignes à chaque itération. Le pas vaut l'inverse de la plus grande valeur propre
du Gram, un choix stable pour cette fonction quadratique. Initialisation à zéro,
critère d'arrêt et nombre d'itérations rendent ce calcul déterministe.
Ce calcul matriciel est adapté à la petite dimension du projet.

### 3. `salary_study/train.py` : orchestration reproductible

`run` charge, sépare et compare les modèles en validation croisée groupée à
cinq plis. `GridSearchCV` choisit les hyperparamètres sur la MAE et réentraîne
chaque candidat sur le train complet. Le gagnant global est celui ayant la plus
petite MAE CV, avant toute évaluation de calibration ou de test.

Le script calibre ensuite son intervalle, évalue les sept références sur le
même test, calcule les diagnostics et SHAP, et exporte les résultats. Les
paramètres et scores par pli, indices de partition, versions, métriques et
empreintes restent consultables. Le modèle final contient le Pipeline entier.
Le modèle n'est pas réentraîné sur calibration ou test.

### 4. `salary_study/uncertainty.py` : intervalles individuels

`conformal_radius` calcule les erreurs absolues sur calibration. Il sélectionne
la valeur au rang `ceil((n+1)*(1-alpha))`. À alpha 0,10, on vise une couverture
marginale de 90 % sous échangeabilité. `prediction_interval` renvoie
`prediction - rayon` et `prediction + rayon`. Le test permet de mesurer la
couverture réellement observée. Si la calibration est trop petite, le code
refuse de prétendre fournir un intervalle fini valable pour ce niveau.

### 5. `salary_study/diagnostics.py` : performances et résidus

`metrics` renvoie R², MAE et RMSE. La MAE est l'erreur absolue moyenne en MAD ;
la RMSE pénalise davantage les grosses erreurs. `bootstrap_metrics` rééchantillonne
le test 1 000 fois pour quantifier l'incertitude de ces métriques, conditionnelle
au modèle figé. Ces IC ne sont pas les intervalles individuels de prédiction.

`residual_diagnostics` analyse `observé - prédit` : normalité (Jarque–Bera),
biais moyen (test t) et variance constante (Breusch–Pagan/Koenker).
Les figures montrent observé/prédit, résidus/prédiction, histogramme et Q-Q plot.
Les tests sont exploratoires et ne certifient pas la validité du marché.

### 6. `salary_study/explain.py` : SHAP

`explain_predictions` transforme les profils avec le prétraitement déjà ajusté.
TreeExplainer explique forêts et boosting ; LinearExplainer explique les modèles
linéaires. Les valeurs SHAP des indicatrices d'une même variable sont additionnées
pour retrouver cinq contributions lisibles. Le programme vérifie que référence
et contributions reconstituent la prédiction. La référence vient du train.
Une attribution SHAP décrit une association du modèle ; les variables liées
comme profil et technologie empêchent une lecture causale simple.

### 7. `salary_study/inference.py` et `app_streamlit.py`

`load_bundle` vérifie l'empreinte modèle/métadonnées et charge le modèle produit
localement. Comme tout pickle, un fichier joblib inconnu ne doit pas être chargé.
`predict` contrôle catégories et domaine d'expérience avant de calculer estimation
et intervalle. La logique est partagée entre tests et application.

L'application affiche trois onglets : résultats et rapport téléchargeable,
simulation avec contributions SHAP locales, puis diagnostics et fiche des données.
Une combinaison profil/technologie absente de l'entraînement est signalée.
Les catégories inconnues et expériences hors domaine sont refusées.

### 8. `salary_study/reporting.py`, documentation et tests

Le rapport client est généré à partir des métriques calculées ; aucune performance
n'est saisie à la main dans ce rapport. Les descriptifs par métier portent sur
le fichier dédupliqué et sont séparés des résultats de test.

`tests/test_study.py` vérifie l'absence de groupes communs entre partitions/plis,
le prétraitement sans apprentissage du test, la convergence NumPy sur une fonction
connue, le rang conformal exact, le schéma, l'additivité SHAP et la validation
d'inférence. Une GitHub Action est préparée pour les exécuter après publication.

## Changements par rapport à l'origine

- Comparaison de cinq familles demandées, plus modèle NumPy et médiane.
- Validation croisée groupée, hyperparamètres documentés et test distinct.
- Calibration séparée, intervalles et couverture par segment.
- Bootstrap, tests statistiques, graphiques des résidus et explications SHAP.
- Application reconstruite autour du même Pipeline que l'entraînement.
- README corrigé, fiche de provenance et rapport client généré.
- Dépendances compatibles Python 3.12 ; suppression de `pickle5` inutile.
- Notebook transformé en entrée pédagogique vers le code commun, sorties
  anciennes effacées pour éviter les performances périmées.
- Ancien `model_salaire_IT_MAROC_2025.pkl` conservé comme artefact historique,
  jamais utilisé par le nouveau parcours. Le nouveau modèle est `artifacts/model.joblib`.

Cette documentation décrit les améliorations apportées au projet initial ; les résultats détaillés sont conservés dans les artefacts d'évaluation.
