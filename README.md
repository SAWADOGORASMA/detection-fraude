# Détection de Fraude dans les Transactions Financières

Projet de fin de Certificat en Intelligence Artificielle — sujet conçu par **Khady KAMA** (Mai 2026).
Classification binaire sur données déséquilibrées à grande échelle, avec gestion du déséquilibre de classes, interprétabilité SHAP et interface de démonstration Gradio.

**Auteur : Rasmané**

> ⚠️ **Usage strictement éducatif.** Ce projet ne constitue en aucun cas un système de détection de fraude en production. Voir la section [Résultat clé](#résultat-clé--à-lire-avant-tout) ci-dessous.

---

## Résultat clé — à lire avant tout

Ce projet n'aboutit pas à un modèle performant, mais à un **diagnostic rigoureux** établi par onze vérifications indépendantes : le dataset Cifer Fraud Detection, tel qu'échantillonné et analysé, ne porte pas la logique causale de fraude nécessaire à une détection fiable au niveau transactionnel (AUC-PR final = 0,049, contre un seuil visé de 0,90 ; rappel = 7,3 %).

Le détail complet de cette démarche — preuves, chiffres, interprétation — est documenté dans [`docs/note_methodologique_revisee.docx`](docs/note_methodologique_revisee.docx) et synthétisé dans [`docs/rapport_technique.docx`](docs/rapport_technique.docx).

---

## Structure du dépôt

```
.
├── data/
│   └── 01_echantillonnage_stratifie.ipynb   # Échantillonnage stratifié (contrainte RAM)
├── eda/
│   ├── 02_qualite_donnees.ipynb             # Contrôle qualité (manquants, doublons, outliers)
│   └── 03_eda_detaillee.ipynb               # Analyse exploratoire (5 axes)
├── features/
│   ├── 04_feature_engineering.ipynb         # Construction des variables (7 catégories)
│   └── 05_split_train_val_test.ipynb        # Split stratifié 70/15/15
├── models/
│   ├── 06_comparaison_modeles.ipynb         # 4 modèles × 3 stratégies de déséquilibre
│   └── 08_optimisation_hyperparametres.ipynb # Recherche bayésienne (Optuna)
├── evaluation/
│   ├── 07_diagnostic_signal_hasard.ipynb    # Diagnostic de l'absence de signal
│   ├── 09_evaluation_finale_test.ipynb      # Évaluation finale sur le test set
│   └── 11_investigation_signal_shap.ipynb   # Investigation du signal résiduel
├── shap_analysis/
│   └── 10_shap_analyse.ipynb                # Interprétabilité SHAP (globale + locale)
├── notebooks/
│   └── projet_complet_fraude.ipynb          # Notebook unique regroupant tout le pipeline
├── docs/
│   ├── note_methodologique_revisee.docx     # Diagnostic complet (11 preuves)
│   ├── rapport_technique.docx               # Rapport technique (5-10 pages)
│   └── presentation_fraude.pptx             # Support de soutenance
├── app.py                         # Interface Gradio de démonstration
├── modele_xgboost_optimise.joblib # Modèle final entraîné (généré par 08_optimisation_hyperparametres.ipynb)
├── requirements.txt               # Dépendances Python
├── .gitignore
└── README.md
```

---

## Installation

**Prérequis** : Python 3.10+, environnement virtuel recommandé.

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Téléchargement des données

Le dataset **Cifer Fraud Detection Dataset-AF** (21 millions de transactions, 14 fichiers CSV, licence Apache 2.0) est disponible sur :
https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF

Télécharge les 14 fichiers et place-les dans un dossier `data/raw/` (non versionné — exclu par `.gitignore`, taille totale de plusieurs Go).

## Exécution du pipeline

Les notebooks sont conçus pour être exécutés **dans l'ordre numéroté** (01 à 11), chacun sauvegardant son résultat pour le suivant. Le notebook `notebooks/projet_complet_fraude.ipynb` regroupe l'intégralité de ces étapes dans un seul fichier, à exécuter cellule par cellule du début à la fin.

| Étape | Notebook | Sortie |
|---|---|---|
| 1 | `data/01_echantillonnage_stratifie.ipynb` | Échantillon stratifié (~1,4M lignes, 100% fraude conservée) |
| 2 | `eda/02_qualite_donnees.ipynb` | Contrôle qualité |
| 3 | `eda/03_eda_detaillee.ipynb` | Analyse exploratoire (5 axes) |
| 4 | `features/04_feature_engineering.ipynb` | Dataset avec features dérivées |
| 5 | `features/05_split_train_val_test.ipynb` | X_train / X_val / X_test / y_train / y_val / y_test |
| 6 | `models/06_comparaison_modeles.ipynb` | Tableau comparatif 12 combinaisons |
| 7 | `evaluation/07_diagnostic_signal_hasard.ipynb` | Diagnostic de l'absence de signal |
| 8 | `models/08_optimisation_hyperparametres.ipynb` | Modèle final optimisé (`.joblib`) |
| 9 | `evaluation/09_evaluation_finale_test.ipynb` | Métriques finales sur le test |
| 10 | `shap_analysis/10_shap_analyse.ipynb` | Graphiques SHAP globaux et locaux |
| 11 | `evaluation/11_investigation_signal_shap.ipynb` | Investigation du signal résiduel |

**Note** : chaque notebook contient un chemin de dossier à adapter à ton environnement local dans sa première cellule de configuration.

## Lancer l'interface de démonstration

Place `modele_xgboost_optimise.joblib` (généré par l'étape 8) au même niveau que `app.py`, puis :

```bash
python app.py
```

Un lien local (`http://127.0.0.1:7860`) s'ouvre dans le navigateur. L'interface permet de saisir les paramètres d'une transaction et d'obtenir une prédiction, un score de risque et une explication SHAP locale.

## Méthodologie résumée

- **Échantillonnage** : stratifié asymétrique (100% fraude + ratio 50:1 sur la non-fraude), contrainte matérielle 8 Go RAM, seed fixée (42) pour la reproductibilité.
- **Feature engineering** : ratios de solde, écarts destinataire, flag d'anomalie de montant, features temporelles, encodage one-hot, features d'agrégation par agent, suppression des identifiants bruts.
- **Modélisation** : Logistic Regression, Random Forest, XGBoost, LightGBM × sous-échantillonnage, SMOTE, coût pondéré.
- **Optimisation** : recherche bayésienne d'hyperparamètres (Optuna, 40 essais).
- **Métrique principale** : AUC-PR (justification détaillée dans le rapport technique).
- **Interprétabilité** : SHAP (TreeExplainer), analyse globale et locale.

## Licence des données

Cifer Fraud Detection Dataset-AF est distribué sous licence Apache 2.0 par CiferAI.

## Contact

Rasmané — Certificat en Intelligence Artificielle
