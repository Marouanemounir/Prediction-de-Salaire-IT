# Étude exploratoire de rémunération IT — Maroc

## Note au client

Ce livrable évalue un prototype de simulation sur le CSV fourni dans le dépôt.
**La provenance des observations n'est pas vérifiée. Ce document ne constitue pas
une enquête représentative du marché marocain, ni une grille salariale validée.**
L'unité MAD/mois vient du dépôt ; brut/net, primes, date réelle de collecte et
type de contrat ne sont pas documentés. Le nom du fichier mentionne 2025 sans
preuve de collecte correspondante.

## Résultats et usage

Le modèle **Lasso** est retenu sur la MAE moyenne en validation croisée à 5 plis
du jeu d'entraînement uniquement. Test final : R² **0.9247**,
MAE **1,143 MAD/mois**, RMSE **1,470 MAD/mois**.
Le R² mesure la variance expliquée ; il ne mesure pas un pourcentage de salaires exacts.
Bootstrap conditionnel au modèle entraîné (1 000 réplications), IC à 95 % :
R² [0.9178 ; 0.9309],
MAE [1,086 ; 1,198] MAD/mois.
Ces intervalles ne mesurent pas l'incertitude liée au réentraînement ni aux biais de collecte.

| Modèle | MAE CV (MAD) | R² test | MAE test (MAD) | RMSE test (MAD) |
|---|---:|---:|---:|---:|
| Lasso | 1,167 | 0.9247 | 1,143 | 1,470 |
| Ridge | 1,171 | 0.9239 | 1,149 | 1,477 |
| Linear Regression | 1,171 | 0.9239 | 1,149 | 1,477 |
| Linear from scratch | 1,171 | 0.9239 | 1,149 | 1,477 |
| Gradient Boosting | 1,272 | 0.9075 | 1,225 | 1,629 |
| Random Forest | 1,372 | 0.9019 | 1,274 | 1,677 |
| Baseline médiane | 4,165 | -0.0197 | 4,324 | 5,408 |

Les scores test des autres modèles sont fournis pour audit ; ils ne servent ni
à choisir le gagnant ni à modifier la grille. Les grilles et scores par pli sont
dans `cv_results.json`. La baseline prédit la médiane d'entraînement.
Les intervalles de confiance des métriques des autres modèles sont dans `metrics.json`.
Des scores proches ne démontrent pas une supériorité statistique entre modèles.

## Fourchettes individuelles

Intervalle conformal nominal à 90 % : prédiction ±
**2,433 MAD/mois**. Couverture observée sur le test :
**92.0 %**, largeur moyenne **4,866 MAD/mois**.
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
49 MAD/mois. Tests exploratoires sur le test final :
- jarque_bera : statistique 139.052, p = 6.39e-31. H0 : Résidus normalement distribués. Rejet de H0 au seuil exploratoire de 5 %.
- bias_t_test : statistique 1.045, p = 0.296. H0 : Erreur moyenne nulle. H0 non rejetée au seuil exploratoire de 5 %.
- breusch_pagan_koenker : statistique 50.686, p = 9.86e-12. H0 : Variance constante par rapport à prédiction et expérience. Rejet de H0 au seuil exploratoire de 5 %.

Seuil descriptif de lecture : 0,05 ; p-values non corrigées pour tests multiples.
Un rejet de normalité n'interdit pas la prédiction ; l'hétéroscédasticité appelle
une lecture prudente de la largeur constante. Sans identifiants ou dates,
l'indépendance des observations n'est pas vérifiable. Les tests ne prouvent pas
la représentativité. Voir `residual_diagnostics.png`, `diagnostics.json` et
`segments_test.csv` (erreur et couverture par profil et niveau d'expérience).

![Diagnostic des résidus sur le test](residual_diagnostics.png)

## Repères descriptifs du fichier

5000 lignes brutes ; 46 doublons exacts exclus ;
4954 lignes exploitées. Salaire observé :
3,000–38,000 MAD/mois ;
médiane 14,000 MAD/mois.
Les repères ci-dessous décrivent toutes les lignes dédupliquées du fichier,
sans pondération, après gel du modèle ; ils ne sont pas des estimations du marché.
L'intervalle Q25–Q75 est descriptif, pas un intervalle de prédiction.

| Profil | Effectif | Médiane MAD/mois | Q25–Q75 MAD/mois |
|---|---:|---:|---:|
| Architecte Solutions | 499 | 16,000 | 12,000–20,000 |
| Business Intelligence | 521 | 13,000 | 10,000–16,000 |
| Chef de Projet IT | 467 | 14,000 | 11,000–18,000 |
| Cloud Engineer | 499 | 15,000 | 12,000–19,000 |
| Data Scientist / IA | 512 | 17,000 | 13,000–21,000 |
| DevOps Engineer | 495 | 15,000 | 12,000–19,000 |
| Développeur Full Stack | 490 | 14,000 | 11,000–18,000 |
| Développeur Mobile | 489 | 13,000 | 10,000–15,000 |
| Ingénieur Cybersécurité | 486 | 16,000 | 12,000–20,000 |
| Ingénieur Réseau | 496 | 11,000 | 9,000–14,000 |

## Explication et décision

Les contributions SHAP sont calculées sur un échantillon déterministe de
200 lignes test, avec 100 lignes d'entraînement comme référence.
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

Graine : 2025 ; tailles : {'train': 2966, 'calibration': 990, 'test': 998}.
Séparation d'environ 60/20/20 par groupes de variables d'entrée identiques.
Les prétraitements sont appris dans chaque pli d'entraînement, puis sur le
jeu d'entraînement final. Ni calibration ni test ne sont utilisés pour ajuster
les modèles ; calibration utilisée uniquement pour le rayon conformal.
Pas de réentraînement après calibration. Indices CSV (base 0) : `split_manifest.csv`.
SHA-256 des données : `85bb7062848038fcc7910b08f737aef640c675b88701b35f9365bc69b4e275bb`.
Paramètres, versions, empreinte du modèle : `metadata.json`.
Provenance et limites détaillées : `../docs/DATA_CARD.md`.
