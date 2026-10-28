# Fiche des données et représentativité

## Éléments vérifiés dans le dépôt

- Fichier : `DATASET_IT_MAROC_2025.csv`, conservé sans modification.
- 5 000 lignes, six colonnes : entreprise, profil, expérience, diplôme,
  technologie principale, salaire. Aucun identifiant de salarié.
- 20 entreprises nommées, 10 profils, 65 technologies, 5 diplômes.
- Expérience : 0 à 15 ans. Salaires : 3 000 à 38 000.
- 46 doublons exacts, exclus des calculs. Des entrées ayant les mêmes cinq
  caractéristiques peuvent avoir des salaires différents ; elles sont conservées
  et regroupées dans le même jeu d'évaluation.
- Tous les salaires sont multiples de 1 000. Cet arrondi ne prouve ni une
  collecte réelle ni une génération synthétique.
- Le notebook initial lit directement le CSV. Il ne contient ni collecte,
  ni générateur, ni citation permettant de remonter à une source primaire.
- Historique consulté : trois commits jusqu'à `49e9886`, avec ajout initial
  `c501283`. Le message initial parle de 3 000 exemples ; aucun fichier de
  collecte ou générateur ne figure dans cette arborescence.
- Le README initial qualifiait les données de « réelles », sans justificatif,
  et annonçait 6 000–65 000 MAD. L'application indiquait 3 000 observations.
  Ces affirmations ont été remplacées par les constats vérifiables.

## Provenance : non vérifiée

Aucune source primaire, URL, protocole d'enquête, licence spécifique du CSV,
date de collecte, définition de population ou méthode d'échantillonnage n'est
fournie dans les fichiers d'origine. Le nom « 2025 » n'établit pas la période
de collecte. Les noms d'entreprises ne prouvent pas leur participation.
L'ancien README mentionne MIT, sans fichier LICENSE ; la licence des données
reste à clarifier avant redistribution ou utilisation commerciale.

Le caractère réel ou synthétique est **indéterminé**. Aucune source externe
trouvée par similarité ne doit être attribuée au CSV sans preuve documentaire.
L'empreinte SHA-256 et les distributions exactes sont calculées à chaque
entraînement dans `artifacts/metadata.json`.

## Ce que le fichier ne permet pas d'établir

Les parts de population des entreprises, villes, genres, tailles d'entreprise,
secteurs, diplômes et métiers ne sont pas connues. Le fichier décrit uniquement
20 entreprises nommées et ne documente pas PME, startups, indépendants ou
secteur public. Aucune pondération ou comparaison à une population de référence
n'est possible. Les technologies et profils sont dépendants.

L'unité MAD/mois provient de la documentation initiale et n'est pas vérifiée par
une fiche de paie. Brut/net, primes, avantages, temps de travail, type de contrat,
ville, date et seniorité réelle ne sont pas documentés. Le plafond observé n'est
pas un plafond du marché. Sans identifiant, des salariés liés ou répétés ne
peuvent pas être reconnus ; supprimer les doublons est un choix conservateur,
pas une preuve d'identité des personnes.

## Périmètre de l'évaluation

Évaluation interne sur une partition déterministe après déduplication, groupée
par les cinq entrées identiques. Elle mesure la prédiction de nouveaux profils
du même fichier, pas de nouvelles entreprises ni de futures années.
Le découpage par groupes et les dépendances éventuelles limitent les hypothèses
IID des tests, du bootstrap et de la garantie conformale ; les résultats restent
exploratoires. Aucune performance externe n'est revendiquée.

## Pièces à demander au propriétaire des données

1. Source d'origine, auteur, autorisation d'usage et licence du CSV.
2. Méthode et dates de collecte, taille initiale, exclusions et consentements.
3. Population cible et plan d'échantillonnage ; distributions de référence.
4. Définition du salaire, brut/net, primes, avantages, contrats et villes.
5. Identifiants pseudonymisés et dates pour traiter répétitions et dépendances.
6. Si synthétique : générateur, hypothèses et graine, avec mention explicite
   « simulation » dans toute présentation.

Avant usage client réel : collecter des données documentées et récentes,
vérifier leur représentativité, puis procéder à une validation externe.
