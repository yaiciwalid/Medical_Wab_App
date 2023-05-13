# DE_project_M2

Nous sommes l'équipe composée de:

**Cherifi Mohamed Ryad**

**Mechouar Fella**

**Yaici Walid**


- Pour le projet de Data Engineering nous allons développer une application web qui traite de différentes pathologies médicales. Le but est que l'utilisateur renseigne différentes données médicales le concernant et grâce à 3 modèles développés, il aura comme résultats:

  * Le premier modèle aura pour but de prédire si le patient est atteint du diabète. 

  * Le deuxième modèle aura pour but de prédire la probabilité qu'il soit atteint d'un cancer du poumon (faible, moyen, élevé).

  * Le troisième modèle concernera le traitement d'images médicales.

- Nous avons crée la structure de notre projet et avons implémentés le preprocessing ainsi que les modèles 1 et 2.

- Pour le preprocessing nous avons utilisé un fichier de configuration et une classe pour chaque dataset de manière a faciliter les futures évolutions et changements dans l'étape de preprocessing.

- Pour les modèles nous avons crée une classe pour chaque modèle.

- Nous avons crée une branche dev a partir de la quelle toutes les autres branches seront crées afin d'éviter de modifier directement sur la branche main.

- Pour chaque partie nous avons crée une ISSUE afin que les commit référencient  des ISSUES et pour chaque issue on associe une branche de dev.

- Nous avons utilisé l'extension Flake8 et le package pre-commit afin de respecter les conventions de codage.

- Pour assurer la qualité du code : on crée une pull request pour chaque issue complétée et aprés review du code par l'équipe on valide la pull request et on merge la branche  sur dev et on supprime la branche.

