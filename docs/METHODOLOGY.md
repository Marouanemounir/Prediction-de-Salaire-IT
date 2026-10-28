# Méthode reproductible

La sélection des modèles suit la [documentation scikit-learn sur les fuites de
données](https://scikit-learn.org/stable/common_pitfalls.html) : séparer avant
d'apprendre le prétraitement ; utiliser un Pipeline dans chaque pli.

1. Valider le schéma, les valeurs finies et le domaine ; exclure les doublons
   exacts sans modifier le CSV source.
2. Grouper les cinq entrées identiques. GroupShuffleSplit avec graine 2025
   réserve environ 20 % au test, puis 25 % du reste à la calibration (graine 2026).
   Les tailles sont approximatives car les groupes ont des tailles différentes.
3. Sur les ~60 % restants, GroupKFold à 5 plis, mélangé avec graine 2025.
   Normalisation de l'expérience et OneHotEncoder appris dans chaque pli.
   Suppression de la première modalité par variable ; catégories inconnues
   encodées à zéro en évaluation et refusées à l'inférence interactive.
4. Grilles fixées dans `models.candidates`. Même CV pour sept références :
   médiane, linéaire sklearn, linéaire NumPy, Ridge, Lasso, forêt et boosting.
   La MAE CV moyenne choisit hyperparamètres et modèle. Les résultats test
   ne déclenchent aucune nouvelle recherche. Entraînement final sur train seul.
5. Calibration du seul gagnant avec les résidus absolus de calibration.
   Ordre statistique `ceil((n_cal+1)*(1-alpha))`, avec erreur explicite si la
   calibration est trop petite. Intervalle symétrique non tronqué. Voir les
   [notes de Ryan Tibshirani, Berkeley](https://statgrades.berkeley.edu/~ryantibs/statlearn-s24/lectures/conformal.pdf).
   Cette garantie nécessite l'échangeabilité des observations ; le découpage
   groupé n'établit pas à lui seul cette hypothèse. Publier la couverture test
   empirique et par segment ; ne pas garantir 90 % par profil.
6. Évaluer une fois sur test : R², MAE, RMSE, bootstrap percentile 95 %
   (1 000 réplications, lignes test rééchantillonnées, graine 2025), couverture,
   largeur et erreurs par segment. Bootstrap conditionnel au modèle figé,
   sans correction des dépendances non documentées.
7. Tests exploratoires : Jarque–Bera (normalité), t de Student à un échantillon
   (résidu moyen nul), [Breusch–Pagan version Koenker](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html)
   (variance selon prédiction et expérience, avec constante). Les p-values sont
   non corrigées. Aucun test d'autocorrélation sur un ordre CSV arbitraire.
8. SHAP : 100 lignes train de référence, jusqu'à 200 lignes test expliquées,
   tirages déterministes. [TreeExplainer interventionnel](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
   pour les arbres ; [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html)
   pour les modèles linéaires. Somme des colonnes encodées par variable d'origine,
   puis moyenne des valeurs absolues agrégées pour le graphique global.
   Additivité vérifiée à 0,05 MAD. Corrélations et encodage de référence influencent
   l'attribution ; aucune interprétation causale.

Le modèle n'est pas réentraîné après calibration ou test. Refaire un entraînement
invaliderait le rayon calibré existant et imposerait une nouvelle calibration.
Les anciens scores du notebook et du pickle ne sont plus utilisés par l'application.
