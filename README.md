# 🎬 CineReco — Système de Recommandation de Films

**Projet ML — Examen L2 Big Data / Informatique | DIT Dakar**

Application Streamlit de recommandation de films basée sur les datasets TMDB (The Movie Database).

---

## 🧠 Modèles de Machine Learning utilisés

| Approche | Algorithme | Justification |
|---|---|---|
| **Content-Based Filtering** | CountVectorizer + Similarité Cosinus | Variables textuelles (genres, acteurs, réalisateur) |
| **Collaborative Filtering** | SVD — TruncatedSVD (sklearn) | Matrice utilisateur-film à variables numériques continues |
| **Hybride** | Combinaison linéaire pondérée (α) | Meilleure couverture et précision |

## 📦 Structure du projet

```
cinereco/
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
├── .streamlit/
│   └── config.toml         # Thème dark + config serveur
├── movies_metadata.csv     # Dataset TMDB (à placer ici)
├── credits.csv             # Dataset crédits (à placer ici)
└── README.md
```

## 🧹 Pipeline de nettoyage des données

1. **3 lignes corrompues** supprimées (valeurs de date dans la colonne `id`)
2. **30 doublons** sur `id` éliminés (première occurrence conservée)
3. **3 188 doublons** de titre résolus
4. Films non-sortis exclus (`status ≠ Released`)
5. `overview` manquant → chaîne vide ; `runtime` → médiane ; `release_date` → drop
6. Films avec `vote_count < 5` filtrés (bruit statistique)

## 🚀 Lancement local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Placer les fichiers CSV dans le même dossier que app.py
# movies_metadata.csv et credits.csv

# Lancer l'application
streamlit run app.py
```

## ☁️ Déploiement sur Streamlit Community Cloud

1. **Fork** ce dépôt sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter votre compte GitHub
4. Sélectionner ce dépôt + `app.py` comme fichier principal
5. ⚠️ **Ajouter les fichiers CSV** : les datasets sont trop volumineux pour GitHub.
   - Option A : utiliser [Git LFS](https://git-lfs.github.com/)
   - Option B : héberger les CSV sur Google Drive / Hugging Face et modifier `load_and_clean()` pour les télécharger via `gdown` ou `huggingface_hub`

### Option B recommandée — téléchargement automatique

Ajouter dans `app.py` avant `load_and_clean()` :

```python
import os, gdown

def download_data():
    if not os.path.exists("movies_metadata.csv"):
        gdown.download(id="VOTRE_GDRIVE_ID_MOVIES", output="movies_metadata.csv")
    if not os.path.exists("credits.csv"):
        gdown.download(id="VOTRE_GDRIVE_ID_CREDITS", output="credits.csv")

download_data()
```

Ajouter `gdown` dans `requirements.txt`.

## 📊 Variables du dataset

### movies_metadata.csv (45 466 films)
| Variable | Type | Description |
|---|---|---|
| `id` | Numérique (int) | Identifiant unique du film |
| `title` | Catégorielle (texte) | Titre du film |
| `genres` | JSON embarqué | Liste des genres |
| `vote_average` | Numérique (float) | Note moyenne /10 |
| `vote_count` | Numérique (int) | Nombre de votes |
| `overview` | Texte | Synopsis |
| `release_date` | Date | Date de sortie |
| `runtime` | Numérique (float) | Durée en minutes |
| `budget` | Numérique (int) | Budget en USD |
| `revenue` | Numérique (int) | Recettes mondiales |
| `popularity` | Numérique (float) | Score de popularité TMDB |
| `original_language` | Catégorielle | Langue originale |
| `status` | Catégorielle | État de sortie |

### credits.csv (45 476 entrées)
| Variable | Type | Description |
|---|---|---|
| `id` | Numérique (int) | Clé de jointure |
| `cast` | JSON embarqué | Liste des acteurs |
| `crew` | JSON embarqué | Équipe technique |

## 🔗 Auteur
DIT Dakar — Département Informatique / Big Data L2
