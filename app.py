"""
Interface Gradio - Démonstration de détection de fraude
=============================================================
Module 4 du sujet (Khady KAMA).

Fonctionnalités :
  - Saisie des paramètres d'une transaction (type, montant, soldes, step)
  - Prédiction (frauduleuse / légitime) avec score de risque
  - Explication SHAP locale (waterfall plot)
  - Indicateur visuel de niveau de risque
  - Avertissement sur l'usage éducatif ET sur la fiabilité limitée du
    modèle (cf. note_methodologique_revisee.docx)

Lancement local : python app.py
Installation : pip install -r requirements.txt

Auteur : Rasmané
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif, juste génération d'images
import matplotlib.pyplot as plt

CHEMIN_MODELE = "modele_xgboost_optimise.joblib"  # doit être placé au même niveau que app.py

modele = joblib.load(CHEMIN_MODELE)
explainer = shap.TreeExplainer(modele)

# Ordre exact des colonnes attendu par le modèle (doit correspondre à
# X_train.csv - copie l'ordre exact affiché par tes scripts précédents)
COLONNES_MODELE = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "isFlaggedFraud", "ratio_solde_orig", "diff_solde_dest",
    "ecart_solde_dest_vs_amount", "flag_montant_egal_solde", "heure_du_jour",
    "jour_simulation", "jour_semaine", "est_nuit", "type_CASH_IN",
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
    "nb_transactions_orig", "montant_moyen_orig", "nb_transactions_dest",
    "montant_moyen_dest",
]

TYPES_TRANSACTION = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

def construire_vecteur_features(step, type_transaction, amount,
                                  oldbalanceOrg, newbalanceOrig,
                                  oldbalanceDest, newbalanceDest):
    """
    Reconstruit un vecteur de features à partir des champs saisis dans
    l'interface, en suivant exactement la même logique que
    feature_engineering.py.

    Limite connue : les features d'agrégation par agent
    (nb_transactions_orig, montant_moyen_orig, nb_transactions_dest,
    montant_moyen_dest) ne peuvent pas être calculées pour une transaction
    isolée saisie manuellement (pas d'historique de l'agent disponible).
    Elles sont fixées à des valeurs par défaut représentant une
    transaction "isolée" (1 transaction, montant moyen = montant de la
    transaction courante), ce qui constitue une approximation. Cette
    limite doit être mentionnée à l'utilisateur dans l'interface.
    """
    # isFlaggedFraud : règle de seuil approximant celle de la simulation
    # d'origine (PaySim : TRANSFER > 200 000). À ajuster si tu connais
    # la règle exacte utilisée par Cifer.
    isFlaggedFraud = int(type_transaction == "TRANSFER" and amount > 200_000)

    ratio_solde_orig = (oldbalanceOrg - newbalanceOrig) / (amount + 1)
    diff_solde_dest = newbalanceDest - oldbalanceDest
    ecart_solde_dest_vs_amount = diff_solde_dest - amount

    tolerance_pct = 0.01
    flag_montant_egal_solde = int(
        oldbalanceOrg > 0
        and (oldbalanceOrg * (1 - tolerance_pct)) <= amount <= (oldbalanceOrg * (1 + tolerance_pct))
    )

    heure_du_jour = int(step % 24)
    jour_simulation = int(step // 24)
    jour_semaine = int(jour_simulation % 7)
    est_nuit = int(heure_du_jour >= 22 or heure_du_jour < 6)

    # Valeurs par défaut pour les features d'agrégation (limite documentée)
    nb_transactions_orig = 1
    montant_moyen_orig = amount
    nb_transactions_dest = 1
    montant_moyen_dest = amount

    donnees = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud,
        "ratio_solde_orig": ratio_solde_orig,
        "diff_solde_dest": diff_solde_dest,
        "ecart_solde_dest_vs_amount": ecart_solde_dest_vs_amount,
        "flag_montant_egal_solde": flag_montant_egal_solde,
        "heure_du_jour": heure_du_jour,
        "jour_simulation": jour_simulation,
        "jour_semaine": jour_semaine,
        "est_nuit": est_nuit,
        "type_CASH_IN": int(type_transaction == "CASH_IN"),
        "type_CASH_OUT": int(type_transaction == "CASH_OUT"),
        "type_DEBIT": int(type_transaction == "DEBIT"),
        "type_PAYMENT": int(type_transaction == "PAYMENT"),
        "type_TRANSFER": int(type_transaction == "TRANSFER"),
        "nb_transactions_orig": nb_transactions_orig,
        "montant_moyen_orig": montant_moyen_orig,
        "nb_transactions_dest": nb_transactions_dest,
        "montant_moyen_dest": montant_moyen_dest,
    }

    X = pd.DataFrame([donnees])[COLONNES_MODELE]  # respecte l'ordre exact
    return X

def determiner_niveau_risque(probabilite):
    if probabilite < 0.20:
        return "Faible", "🟢"
    elif probabilite < 0.40:
        return "Modéré", "🟡"
    elif probabilite < 0.60:
        return "Élevé", "🟠"
    else:
        return "Critique", "🔴"

def predire_transaction(step, type_transaction, amount,
                         oldbalanceOrg, newbalanceOrig,
                         oldbalanceDest, newbalanceDest):

    X = construire_vecteur_features(
        step, type_transaction, amount,
        oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    )

    probabilite = modele.predict_proba(X)[0, 1]
    prediction = "FRAUDULEUSE" if probabilite >= 0.5 else "LÉGITIME"
    niveau, icone = determiner_niveau_risque(probabilite)

    texte_resultat = f"""
### {icone} Prédiction : {prediction}

**Score de risque (probabilité de fraude) : {probabilite*100:.1f}%**

**Niveau de risque : {niveau}**

---
⚠️ **Rappel important** : d'après l'analyse méthodologique de ce projet, ce modèle ne dispose que d'un
signal prédictif très faible sur ce dataset (AUC-PR ≈ 0,05, contre un
seuil de fiabilité opérationnelle attendu de 0,90). **Cette prédiction ne
doit pas être interprétée comme fiable** — elle est fournie à titre de
démonstration technique du pipeline uniquement.
"""

    # Explication SHAP locale
    valeurs_shap = explainer.shap_values(X)
    explication = shap.Explanation(
        values=valeurs_shap[0],
        base_values=explainer.expected_value,
        data=X.iloc[0].values,
        feature_names=X.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explication, max_display=10, show=False)
    plt.tight_layout()

    return texte_resultat, fig

with gr.Blocks(title="Démo - Détection de fraude - Rasmané (usage éducatif)") as app:

    gr.Markdown("""
    # 🔍 Démonstrateur de détection de fraude financière

    **Projet réalisé par SAWADOGO Rasmané** dans le cadre du Certificat en Intelligence Artificielle,
    sur le dataset Cifer Fraud Detection.
    """)

    gr.Markdown("""
    > ⚠️ **AVERTISSEMENT — À LIRE AVANT UTILISATION**
    >
    > 1. **Usage strictement éducatif.** Cette interface ne constitue en
    >    aucun cas un système de détection de fraude en production.
    > 2. **Fiabilité limitée du modèle.** L'analyse exploratoire approfondie
    >    de ce projet a démontré que le dataset utilisé ne contient qu'un
    >    signal prédictif très faible (AUC-PR ≈ 0,05 sur un seuil visé de
    >    0,90 ; rappel ≈ 7 % en pratique). **Les prédictions affichées ici
    >    ne doivent pas être considérées comme fiables.**
    > 3. Les features d'agrégation par agent (historique du client) ne
    >    peuvent pas être calculées pour une transaction saisie isolément
    >    et sont approximées — cela peut affecter davantage la prédiction.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Paramètres de la transaction")
            step_input = gr.Number(label="Step (heure écoulée depuis le début — 0 à 743)", value=100, precision=0)
            type_input = gr.Dropdown(choices=TYPES_TRANSACTION, label="Type de transaction", value="TRANSFER")
            amount_input = gr.Number(label="Montant (amount)", value=50000)
            oldbalanceOrg_input = gr.Number(label="Solde émetteur avant (oldbalanceOrg)", value=100000)
            newbalanceOrig_input = gr.Number(label="Solde émetteur après (newbalanceOrig)", value=50000)
            oldbalanceDest_input = gr.Number(label="Solde destinataire avant (oldbalanceDest)", value=0)
            newbalanceDest_input = gr.Number(label="Solde destinataire après (newbalanceDest)", value=50000)
            bouton_predire = gr.Button("🔎 Analyser la transaction", variant="primary")

        with gr.Column():
            gr.Markdown("### Résultat")
            resultat_texte = gr.Markdown()
            gr.Markdown("### Explication SHAP locale (contribution de chaque variable)")
            resultat_graphique = gr.Plot()

    bouton_predire.click(
        fn=predire_transaction,
        inputs=[step_input, type_input, amount_input, oldbalanceOrg_input,
                newbalanceOrig_input, oldbalanceDest_input, newbalanceDest_input],
        outputs=[resultat_texte, resultat_graphique]
    )

    gr.Markdown("""
    ---
    *Modèle : XGBoost optimisé (Optuna). Dataset : Cifer Fraud Detection
    Dataset-AF (échantillon stratifié). Voir le rapport technique complet
    pour le détail du diagnostic de performance.*

    **Auteur : SAWADOGO Rasmané**
    """)


if __name__ == "__main__":
    app.launch()
