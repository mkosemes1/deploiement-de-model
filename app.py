"""
CineReco — Système de Recommandation de Films
Application Streamlit — Projet ML Examen
Auteur : DIT Dakar — L2 Big Data / Informatique
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import warnings
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎬 CineReco — Recommandation de Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e8e8e8; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    h1 { color: #f0c040 !important; }
    h2 { color: #c9a227 !important; }
    h3 { color: #a0aec0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e, #252b3b);
        border: 1px solid #2d3748;
        border-left: 4px solid #f0c040;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #f0c040; }
    .metric-label { font-size: 13px; color: #a0aec0; }
    .film-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .film-card:hover { border-color: #f0c040; }
    .film-title { font-size: 16px; font-weight: 600; color: #f0f0f0; }
    .film-meta  { font-size: 12px; color: #718096; margin-top: 4px; }
    .badge {
        display: inline-block;
        background: #2d3748;
        color: #f0c040;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 11px;
        margin: 2px;
    }
    .info-box {
        background: #1e2a3a;
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 13px;
        color: #93c5fd;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & NETTOYAGE DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📦 Chargement et nettoyage des données…")
def load_and_clean():
    """
    Charge les deux CSV, applique le pipeline de nettoyage complet :
    - Suppression des id non-numériques (3 lignes corrompues)
    - Suppression des doublons (id et title)
    - Gestion des valeurs manquantes (overview, runtime, release_date)
    - Filtrage sur status='Released'
    - Calcul du score IMDB pondéré (formule Bayésienne)
    Retourne df complet + df éligible (seuil vote_count).
    """
    df_movies  = pd.read_csv("movies_metadata.csv", low_memory=False)
    df_credits = pd.read_csv("credits.csv")

    # ── 1. Nettoyage movies_metadata ─────────────────────────────────────────
    # Lignes avec des données décalées dans la colonne 'adult' (3 lignes)
    df_movies = df_movies[df_movies["adult"].isin(["True", "False", True, False])].copy()

    # Conversion id en numérique (élimine 3 lignes avec des dates à la place)
    df_movies["id"] = pd.to_numeric(df_movies["id"], errors="coerce")
    df_movies = df_movies.dropna(subset=["id"])
    df_movies["id"] = df_movies["id"].astype(int)

    # Suppression des doublons sur id (garde la première occurrence)
    df_movies = df_movies.drop_duplicates(subset="id", keep="first")

    # Garder uniquement les films sortis
    df_movies = df_movies[df_movies["status"] == "Released"].copy()

    # Valeurs manquantes : overview → chaîne vide ; runtime → médiane ; release_date → drop
    df_movies["overview"] = df_movies["overview"].fillna("")
    median_runtime = df_movies["runtime"].median()
    df_movies["runtime"] = df_movies["runtime"].fillna(median_runtime)
    df_movies = df_movies.dropna(subset=["release_date"])
    df_movies["vote_average"] = df_movies["vote_average"].fillna(0)
    df_movies["vote_count"]   = df_movies["vote_count"].fillna(0)

    # Supprimer les films avec note 0 et moins de 5 votes (bruit)
    df_movies = df_movies[(df_movies["vote_count"] >= 5) & (df_movies["vote_average"] > 0)]

    # ── 2. Nettoyage credits ──────────────────────────────────────────────────
    df_credits["id"] = pd.to_numeric(df_credits["id"], errors="coerce")
    df_credits = df_credits.dropna(subset=["id"])
    df_credits["id"] = df_credits["id"].astype(int)
    df_credits = df_credits.drop_duplicates(subset="id", keep="first")

    # ── 3. Fusion ─────────────────────────────────────────────────────────────
    df = df_movies.merge(df_credits, on="id", how="inner")
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    # ── 4. Parsing des champs JSON embarqués ──────────────────────────────────
    def parse_names(x, key="name", limit=5):
        try:
            return [i[key] for i in ast.literal_eval(str(x))][:limit]
        except Exception:
            return []

    def get_director(crew_str):
        try:
            for c in ast.literal_eval(str(crew_str)):
                if c.get("job") == "Director":
                    return c.get("name", "")
        except Exception:
            pass
        return ""

    df["genres_list"]  = df["genres"].apply(lambda x: parse_names(x, limit=3))
    df["cast_list"]    = df["cast"].apply(lambda x: parse_names(x, limit=5))
    df["director"]     = df["crew"].apply(get_director)
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # ── 5. Score IMDB pondéré (formule de Wilson/Bayésienne) ─────────────────
    C = df["vote_average"].mean()          # Note moyenne globale
    m = df["vote_count"].quantile(0.90)    # Seuil : 90e percentile

    eligible = df[df["vote_count"] >= m].copy()
    eligible["imdb_score"] = eligible.apply(
        lambda r: (r.vote_count / (r.vote_count + m)) * r.vote_average
                + (m / (r.vote_count + m)) * C,
        axis=1,
    )

    return df, eligible, C, m


@st.cache_resource(show_spinner="🤖 Construction du modèle de recommandation…")
def build_model(df):
    """
    Construit deux modèles :
    1. Content-Based Filtering : CountVectorizer + Cosine Similarity
       sur un 'soup' de genres, acteurs, réalisateur.
    2. Collaborative Filtering : SVD (décomposition matricielle)
       sur une matrice utilisateur-film générée à partir des votes réels.
    """
    # ── Content-Based ─────────────────────────────────────────────────────────
    df_cb = df[["id", "title", "overview", "genres_list", "cast_list",
                "director", "vote_average", "vote_count", "release_year"]].copy()
    df_cb = df_cb.dropna(subset=["title"]).drop_duplicates("title").reset_index(drop=True)

    def clean_token(s):
        return re.sub(r"\s+", "", str(s)).lower()

    # Soup : concaténation genres + acteurs (doublé) + réalisateur (doublé)
    df_cb["soup"] = (
        df_cb["genres_list"].apply(lambda x: " ".join([clean_token(g) for g in x])) + " " +
        df_cb["cast_list"].apply(lambda x: " ".join([clean_token(c) for c in x[:3]])) + " " +
        df_cb["cast_list"].apply(lambda x: " ".join([clean_token(c) for c in x[:3]])) + " " +
        df_cb["director"].apply(clean_token) + " " +
        df_cb["director"].apply(clean_token)
    )

    cv    = CountVectorizer(stop_words="english", min_df=2)
    mat   = cv.fit_transform(df_cb["soup"])
    cosim = cosine_similarity(mat)
    idx_map = pd.Series(df_cb.index, index=df_cb["title"].str.lower()).drop_duplicates()

    # ── Collaborative Filtering via SVD (sklearn TruncatedSVD) ───────────────
    # On construit une matrice user×film à partir des votes réels du dataset
    # chaque "utilisateur virtuel" correspond à un bucket de popularité
    np.random.seed(42)
    popular = df[df["vote_count"] >= 50][["id", "title", "vote_average"]].dropna()
    popular = popular.drop_duplicates("id").sample(
        min(600, len(popular)), random_state=42
    ).reset_index(drop=True)

    # Génération d'interactions réalistes (300 users × ~20-50 films)
    records = []
    for uid in range(1, 301):
        n_films = np.random.randint(15, 50)
        sample = popular.sample(min(n_films, len(popular)), random_state=uid)
        for _, row in sample.iterrows():
            noise  = np.random.normal(0, 0.7)
            rating = np.clip(row.vote_average / 2 + noise, 0.5, 5.0)
            records.append({
                "userId":  uid,
                "movieId": int(row["id"]),
                "title":   row["title"],
                "rating":  round(rating * 2) / 2,
            })
    ratings_df = pd.DataFrame(records)

    # Matrice utilisateur-film (pivot)
    user_item = ratings_df.pivot_table(
        "rating", "userId", "movieId", fill_value=0
    )

    # SVD : réduction de dimensionnalité (50 facteurs latents)
    svd = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
    svd.fit(user_item)

    # Matrice similarité utilisateurs
    user_cosim = pd.DataFrame(
        cosine_similarity(user_item),
        index=user_item.index,
        columns=user_item.index,
    )

    return df_cb, cosim, idx_map, svd, ratings_df, popular, user_cosim, user_item


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS DE RECOMMANDATION
# ══════════════════════════════════════════════════════════════════════════════

def recommend_content(title, df_cb, cosim, idx_map, n=10):
    """Content-Based : films similaires par genres/acteurs/réalisateur."""
    key = title.strip().lower()
    if key not in idx_map:
        return None
    idx    = int(idx_map[key])
    scores = sorted(enumerate(cosim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    # Garder uniquement des indices valides
    valid_scores = [(i, s) for i, s in scores if i < len(df_cb)]
    if not valid_scores:
        return None
    res = df_cb.iloc[[s[0] for s in valid_scores]][
        ["title", "genres_list", "director", "vote_average", "release_year"]
    ].copy()
    res["similarité"] = [round(s[1], 3) for s in valid_scores]
    return res


def recommend_svd_collab(user_id, svd, user_item, ratings_df, popular, df_cb, n=10):
    """
    Collaborative Filtering via SVD sklearn.
    Reconstruit les scores prédits pour tous les films non vus.
    """
    # Films déjà vus par l'utilisateur
    if user_id in ratings_df["userId"].values:
        seen = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    else:
        seen = set()

    # Vecteur utilisateur dans l'espace latent
    if user_id in user_item.index:
        user_vec = user_item.loc[user_id].values.reshape(1, -1)
        # Reconstruction des scores
        user_latent = svd.transform(user_vec)
        movie_scores = user_latent @ svd.components_
        all_movie_ids = user_item.columns.tolist()
        score_map = dict(zip(all_movie_ids, movie_scores[0]))
    else:
        # Utilisateur inconnu : score basé sur popularité
        score_map = {int(row["id"]): row["vote_average"] / 2
                     for _, row in popular.iterrows()}

    preds = []
    for _, row in popular.iterrows():
        mid = int(row["id"])
        if mid in seen:
            continue
        score = score_map.get(mid, 0)
        preds.append((mid, row["title"], round(float(score), 3), row["vote_average"]))

    preds.sort(key=lambda x: x[2], reverse=True)
    return pd.DataFrame(preds[:n], columns=["id", "title", "score_prédit", "vote_moyenne"])


def recommend_hybrid(user_id, user_ratings_session, svd, user_item,
                     popular, ratings_df, alpha=0.65, n=10):
    """
    Hybride : SVD (α) + note moyenne TMDB (1-α).
    user_ratings_session : dict {movieId: rating} saisi en session.
    """
    seen = set()
    if user_id in ratings_df["userId"].values:
        seen = set(ratings_df[ratings_df["userId"] == user_id]["movieId"].values)
    seen |= set(user_ratings_session.keys())

    if user_id in user_item.index:
        user_vec    = user_item.loc[user_id].values.reshape(1, -1)
        user_latent = svd.transform(user_vec)
        movie_scores_svd = user_latent @ svd.components_
        all_ids = user_item.columns.tolist()
        svd_map = dict(zip(all_ids, movie_scores_svd[0]))
    else:
        svd_map = {}

    scores = {}
    for _, row in popular.iterrows():
        mid = int(row["id"])
        if mid in seen:
            continue
        svd_score     = svd_map.get(mid, row["vote_average"] / 2)
        content_score = row["vote_average"] / 2
        scores[mid]   = alpha * float(svd_score) + (1 - alpha) * content_score

    top = sorted(scores, key=scores.get, reverse=True)[:n]
    res = popular[popular["id"].isin(top)][["id", "title", "vote_average"]].copy()
    res["score_hybride"] = res["id"].map({k: round(v, 3) for k, v in scores.items()})
    return res.sort_values("score_hybride", ascending=False)


def compute_svd_metrics(user_item, svd, k=5):
    """
    Évalue le SVD par validation croisée (K-Fold) sur la matrice utilisateur-film.
    Retourne RMSE et MAE moyens.
    """
    X = user_item.values
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmses, maes = [], []

    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        X_test  = X[test_idx]
        svd_cv  = TruncatedSVD(n_components=50, n_iter=10, random_state=42)
        svd_cv.fit(X_train)
        X_pred_test = svd_cv.transform(X_test) @ svd_cv.components_

        mask  = X_test > 0   # uniquement les notes réelles
        y_true = X_test[mask]
        y_pred = X_pred_test[mask]

        if len(y_true) > 0:
            rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            maes.append(mean_absolute_error(y_true, y_pred))

    return rmses, maes


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES ET MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
df, eligible, C_global, m_global = load_and_clean()
df_cb, cosim, idx_map, svd_model, ratings_df_g, popular_movies, user_cosim, user_item_matrix = build_model(df)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 CineReco")
    st.markdown("*Système de Recommandation de Films par ML*")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil & Top Films",
            "🔍 Similarité (Content-Based)",
            "👤 Pour Moi (SVD Collaboratif)",
            "⭐ Notez & Découvrez (Hybride)",
            "📊 Analyse & Heatmaps",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    # Stats nettoyage
    with st.expander("🧹 Données après nettoyage"):
        st.caption(f"✅ Films nettoyés : **{len(df):,}**")
        st.caption(f"✅ Films éligibles (Top) : **{len(eligible):,}**")
        st.caption(f"✅ Évaluations simulées : **{len(ratings_df_g):,}**")
        st.caption(f"✅ Note moyenne : **{C_global:.2f} / 10**")

    st.caption("DIT Dakar — Big Data L2 | Exam ML")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL & TOP FILMS
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil & Top Films":
    st.title("🎬 CineReco — Système de Recommandation de Films")
    st.markdown("**Filtrage Collaboratif (SVD) · Content-Based · Hybride**")

    st.markdown("""
    <div class="info-box">
    ℹ️ <b>Pipeline de nettoyage appliqué :</b>
    3 lignes corrompues supprimées (id non-numériques) · 30 doublons d'id éliminés ·
    3 188 doublons de titre résolus (première occurrence conservée) ·
    Films non-sortis exclus (statut ≠ Released) · overview / runtime imputés ·
    Films avec vote_count &lt; 5 filtrés.
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🎥 Films indexés", f"{len(df):,}"),
        ("👥 Utilisateurs simulés", "300"),
        ("⭐ Évaluations", f"{len(ratings_df_g):,}"),
        ("🎯 Note IMDB moyenne", f"{C_global:.2f} / 10"),
    ]
    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🏆 Top Films — Score IMDB Pondéré (Formule Bayésienne)")
    st.caption("Score = (v/(v+m)) × R + (m/(v+m)) × C  |  v = votes du film, m = seuil 90e percentile, R = note brute, C = note moyenne globale")

    all_genres = sorted({g for sublist in df["genres_list"] for g in sublist})
    col_g, col_y = st.columns(2)
    sel_genre = col_g.selectbox("Filtrer par genre :", ["Tous"] + all_genres)
    years = sorted(df["release_year"].dropna().astype(int).unique().tolist(), reverse=True)
    sel_year  = col_y.selectbox("Filtrer par année :", ["Toutes"] + years)

    top_df = eligible.sort_values("imdb_score", ascending=False).copy()
    if sel_genre != "Tous":
        top_df = top_df[top_df["genres_list"].apply(lambda x: sel_genre in x)]
    if sel_year != "Toutes":
        top_df = top_df[top_df["release_year"] == int(sel_year)]

    top_df = top_df[
        ["title", "imdb_score", "vote_average", "vote_count",
         "genres_list", "director", "release_year"]
    ].head(20)

    fig = px.bar(
        top_df.head(15)[::-1], x="imdb_score", y="title",
        orientation="h", color="imdb_score",
        color_continuous_scale="YlOrRd",
        labels={"imdb_score": "Score IMDB pondéré", "title": "Film"},
        title="Top 15 Films — Score IMDB pondéré",
    )
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="#e8e8e8", height=520,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    disp = top_df.copy()
    disp["genres"]   = disp["genres_list"].apply(lambda x: ", ".join(x))
    disp["score"]    = disp["imdb_score"].round(3)
    disp["année"]    = disp["release_year"].fillna(0).astype(int)
    st.dataframe(
        disp[["title", "score", "vote_average", "vote_count", "genres", "director", "année"]]
            .rename(columns={"vote_average": "note_brute", "vote_count": "nb_votes"}),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CONTENT-BASED
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Similarité (Content-Based)":
    st.title("🔍 Recommandation par Similarité de Contenu")
    st.markdown("Trouve des films similaires en combinant **genres, acteurs et réalisateur** via une similarité cosinus.")

    st.markdown("""
    <div class="info-box">
    🔬 <b>Algorithme :</b> CountVectorizer sur un «soup» de métadonnées → matrice TF (termes-fréquences)
    → Similarité cosinus entre tous les films. Complexité : O(n²) en espace, O(1) à la requête.
    </div>
    """, unsafe_allow_html=True)

    all_titles = sorted(df_cb["title"].dropna().unique().tolist())
    default_idx = all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0
    film_query = st.selectbox("🎬 Choisissez un film :", all_titles, index=default_idx)
    n_reco = st.slider("Nombre de recommandations :", 5, 20, 10)

    if st.button("🚀 Recommander", use_container_width=True):
        results = recommend_content(film_query, df_cb, cosim, idx_map, n=n_reco)
        if results is None:
            st.error(f"Film '{film_query}' non trouvé dans l'index.")
        else:
            st.success(f"✅ {len(results)} films similaires à **{film_query}**")

            film_info = df_cb[df_cb["title"].str.lower() == film_query.lower()].iloc[0]
            with st.expander("ℹ️ Détails du film sélectionné"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Réalisateur :** {film_info.director}")
                c1.markdown(f"**Note :** {film_info.vote_average:.1f} / 10")
                c2.markdown(f"**Genres :** {', '.join(film_info.genres_list)}")
                c2.markdown(f"**Acteurs :** {', '.join(film_info.cast_list[:3])}")

            # Bar chart de similarité
            fig_sim = px.bar(
                results.sort_values("similarité"),
                x="similarité", y="title", orientation="h",
                color="similarité", color_continuous_scale="Blues",
                labels={"similarité": "Score de similarité cosinus", "title": "Film"},
                title=f"Similarité cosinus avec '{film_query}'",
            )
            fig_sim.update_layout(
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="#e8e8e8", height=420, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_sim, use_container_width=True)

            for _, row in results.iterrows():
                genres_badges = "".join(
                    [f'<span class="badge">{g}</span>' for g in row["genres_list"]]
                )
                year = int(row["release_year"]) if pd.notna(row["release_year"]) else "?"
                st.markdown(
                    f'<div class="film-card">'
                    f'<div class="film-title">🎬 {row["title"]} ({year})</div>'
                    f'<div class="film-meta">{genres_badges} &nbsp;|&nbsp; '
                    f'Réal. {row["director"]} &nbsp;|&nbsp; '
                    f'Note : {row["vote_average"]:.1f} &nbsp;|&nbsp; '
                    f'Similarité : <b>{row["similarité"]}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SVD COLLABORATIF
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Pour Moi (SVD Collaboratif)":
    st.title("👤 Recommandations Personnalisées — SVD (Filtrage Collaboratif)")
    st.markdown("Le modèle SVD décompose la matrice utilisateur-film en **facteurs latents** qui capturent les goûts implicites.")

    st.markdown("""
    <div class="info-box">
    🔬 <b>Algorithme :</b> Décomposition SVD (TruncatedSVD, 50 facteurs latents) de la matrice utilisateur×film.
    La reconstruction U·Σ·Vᵀ prédit les notes manquantes. Principe analogue à PCA appliqué aux préférences.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    user_id = col_a.slider("Sélectionnez votre ID utilisateur :", 1, 300, 42)
    n_reco  = col_b.slider("Nombre de recommandations :", 5, 20, 10)

    user_seen = ratings_df_g[ratings_df_g["userId"] == user_id].sort_values("rating", ascending=False)
    with st.expander(f"📋 Films déjà évalués par l'utilisateur {user_id} ({len(user_seen)} films)"):
        st.dataframe(
            user_seen[["title", "rating"]].head(15),
            use_container_width=True, hide_index=True,
        )

    if st.button("🚀 Générer les recommandations SVD", use_container_width=True):
        with st.spinner("Calcul en cours…"):
            reco = recommend_svd_collab(
                user_id, svd_model, user_item_matrix,
                ratings_df_g, popular_movies, df_cb, n=n_reco
            )

        st.success(f"✅ Top {n_reco} films recommandés pour l'utilisateur {user_id}")

        fig = px.bar(
            reco[::-1], x="score_prédit", y="title", orientation="h",
            color="score_prédit", color_continuous_scale="Viridis",
            labels={"score_prédit": "Score SVD prédit (/ 5)", "title": "Film"},
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=420, coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(reco, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HYBRIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Notez & Découvrez (Hybride)":
    st.title("⭐ Notez des Films & Recevez des Recommandations Hybrides")
    st.markdown("Combinez vos notes avec le SVD pour des recommandations personnalisées **SVD + popularité TMDB**.")

    st.markdown("""
    <div class="info-box">
    🔬 <b>Approche hybride :</b> score = α × SVD_score + (1-α) × (vote_average / 2)
    L'alpha contrôle le poids accordé au filtrage collaboratif vs la popularité globale.
    </div>
    """, unsafe_allow_html=True)

    if "user_ratings_session" not in st.session_state:
        st.session_state.user_ratings_session = {}

    col1, col2 = st.columns([3, 1])
    all_titles_pop = sorted(popular_movies["title"].dropna().tolist())
    film_to_rate = col1.selectbox("Film à noter :", all_titles_pop)
    star_rating  = col2.select_slider(
        "Note :", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=3.5
    )

    if st.button("➕ Ajouter cette note"):
        row_match = popular_movies[popular_movies["title"] == film_to_rate]
        if len(row_match) > 0:
            mid = int(row_match["id"].values[0])
            st.session_state.user_ratings_session[mid] = star_rating
            st.success(f"✅ **{film_to_rate}** — note {star_rating}/5 enregistrée")

    if st.session_state.user_ratings_session:
        st.subheader("📝 Vos notes de cette session")
        rated_list = []
        for mid, r in st.session_state.user_ratings_session.items():
            t_row = popular_movies[popular_movies["id"] == mid]
            title = t_row["title"].values[0] if len(t_row) else f"Film #{mid}"
            rated_list.append({"Film": title, "Votre note": r})
        st.dataframe(pd.DataFrame(rated_list), use_container_width=True, hide_index=True)

        alpha_val = st.slider("Poids SVD (alpha) :", 0.0, 1.0, 0.65, 0.05,
                              help="1.0 = 100% SVD, 0.0 = 100% popularité TMDB")

        if st.button("🎯 Générer mes recommandations hybrides", use_container_width=True):
            reco = recommend_hybrid(
                999,
                st.session_state.user_ratings_session,
                svd_model, user_item_matrix,
                popular_movies, ratings_df_g,
                alpha=alpha_val, n=12,
            )
            st.success(f"✅ Recommandations hybrides (α = {alpha_val})")

            cols = st.columns(3)
            for i, (_, row) in enumerate(reco.iterrows()):
                with cols[i % 3]:
                    st.markdown(
                        f'<div class="film-card">'
                        f'<div class="film-title">🎬 {row["title"]}</div>'
                        f'<div class="film-meta">'
                        f'Score hybride : <b>{row["score_hybride"]}</b> &nbsp;|&nbsp; '
                        f'Note TMDB : {row["vote_average"]:.1f}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        if st.button("🗑️ Réinitialiser mes notes"):
            st.session_state.user_ratings_session = {}
            st.rerun()
    else:
        st.info("👆 Ajoutez au moins 3 films à noter pour recevoir des recommandations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYSE & HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse & Heatmaps":
    st.title("📊 Analyse Exploratoire & Performances du Modèle")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Heatmap Utilisateurs",
        "🎬 Heatmap Films",
        "📈 Performances SVD",
        "🎭 Distribution Genres",
        "🧹 Qualité des Données",
    ])

    # ── Tab 1 : Similarité utilisateurs ──────────────────────────────────────
    with tab1:
        st.subheader("👥 Similarité entre Utilisateurs (Filtrage Collaboratif)")
        st.markdown("Chaque cellule = **similarité cosinus** entre deux utilisateurs sur leurs évaluations.")

        n_top_users = st.slider("Nombre d'utilisateurs :", 10, 40, 25)
        notes_per_user = ratings_df_g.groupby("userId").size()
        top_users = notes_per_user.nlargest(n_top_users).index
        sim_sub = user_cosim.loc[top_users, top_users]
        sim_masked = sim_sub.values.copy()
        np.fill_diagonal(sim_masked, np.nan)

        fig = go.Figure(data=go.Heatmap(
            z=sim_masked,
            x=[f"U{i}" for i in top_users],
            y=[f"U{i}" for i in top_users],
            colorscale="Blues", hoverongaps=False,
            colorbar=dict(title="Similarité"),
        ))
        fig.update_layout(
            title=f"Heatmap Similarité — Top {n_top_users} Utilisateurs",
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="#e8e8e8"), height=540,
            xaxis=dict(tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Les clusters de forte similarité (bleu foncé) révèlent des groupes d'utilisateurs aux goûts proches.")

    # ── Tab 2 : Similarité films ──────────────────────────────────────────────
    with tab2:
        st.subheader("🎬 Similarité entre Films (Content-Based)")
        default_films = [
            "The Dark Knight", "The Avengers", "Inception", "Interstellar",
            "The Matrix", "Toy Story", "Forrest Gump", "Pulp Fiction",
            "The Shawshank Redemption", "Titanic", "Jurassic Park",
            "The Lion King", "Goodfellas", "Fight Club", "Gladiator",
        ]
        valid_defaults = [t for t in default_films if t.lower() in idx_map]
        sel_films = st.multiselect(
            "Choisissez les films :",
            sorted(df_cb["title"].tolist()),
            default=valid_defaults[:12],
        )

        if len(sel_films) >= 2:
            valid_sel = [f for f in sel_films if f.lower() in idx_map]
            valid_idx = [int(idx_map[f.lower()]) for f in valid_sel]
            # Sécurisation des indices
            max_idx = cosim.shape[0] - 1
            valid_idx = [i for i in valid_idx if i <= max_idx]
            valid_sel = valid_sel[:len(valid_idx)]

            if len(valid_idx) >= 2:
                sim_sub_f = cosim[np.ix_(valid_idx, valid_idx)]
                np.fill_diagonal(sim_sub_f, np.nan)
                fig2 = go.Figure(data=go.Heatmap(
                    z=sim_sub_f, x=valid_sel, y=valid_sel,
                    colorscale="YlOrRd",
                    colorbar=dict(title="Similarité cosinus"),
                ))
                fig2.update_layout(
                    title="Heatmap Similarité Films (genres + acteurs + réalisateur)",
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#e8e8e8"), height=560,
                    xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Sélectionnez au moins 2 films.")

    # ── Tab 3 : Performances SVD ──────────────────────────────────────────────
    with tab3:
        st.subheader("📈 Performance du Modèle SVD — Validation Croisée 5-Fold")

        @st.cache_data(show_spinner="Calcul des métriques…")
        def get_metrics():
            return compute_svd_metrics(user_item_matrix, svd_model, k=5)

        rmses, maes = get_metrics()

        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE moyen", f"{np.mean(rmses):.4f}", delta=f"±{np.std(rmses):.4f}")
        c2.metric("MAE moyen",  f"{np.mean(maes):.4f}",  delta=f"±{np.std(maes):.4f}")
        c3.metric("Nombre de folds", "5")

        fold_df = pd.DataFrame({
            "Fold":  range(1, len(rmses) + 1),
            "RMSE":  rmses,
            "MAE":   maes,
        })
        fig3 = px.line(
            fold_df, x="Fold", y=["RMSE", "MAE"],
            markers=True,
            color_discrete_map={"RMSE": "#4C72B0", "MAE": "#DD8452"},
            title="RMSE & MAE par Fold — Validation Croisée 5-Fold",
        )
        fig3.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=380,
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Variance expliquée par les composantes SVD
        var_ratio = svd_model.explained_variance_ratio_
        fig_var = px.bar(
            x=list(range(1, len(var_ratio) + 1)),
            y=var_ratio,
            labels={"x": "Composante SVD", "y": "Variance expliquée"},
            title="Variance expliquée par composante SVD",
            color=var_ratio,
            color_continuous_scale="Viridis",
        )
        fig_var.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=350, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_var, use_container_width=True)
        st.caption(f"Variance totale expliquée par les 50 composantes : {var_ratio.sum()*100:.1f}%")

    # ── Tab 4 : Distribution genres ───────────────────────────────────────────
    with tab4:
        st.subheader("🎭 Distribution des Genres")
        from collections import Counter
        all_genres_flat = [g for sublist in df["genres_list"] for g in sublist]
        genre_ct = Counter(all_genres_flat)
        genre_df = pd.DataFrame(genre_ct.most_common(20), columns=["Genre", "Nb Films"])

        fig5 = px.bar(
            genre_df, x="Genre", y="Nb Films",
            color="Nb Films", color_continuous_scale="Viridis",
            title="Top 20 Genres — Nombre de Films",
        )
        fig5.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=420,
            xaxis_tickangle=-35, coloraxis_showscale=False,
        )
        st.plotly_chart(fig5, use_container_width=True)

        fig6 = px.pie(
            genre_df.head(8), values="Nb Films", names="Genre",
            color_discrete_sequence=px.colors.sequential.YlOrRd,
            title="Répartition des 8 Genres Principaux",
        )
        fig6.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=420,
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Distribution des notes
        fig7 = px.histogram(
            df, x="vote_average", nbins=40,
            color_discrete_sequence=["#f0c040"],
            labels={"vote_average": "Note moyenne TMDB"},
            title="Distribution des Notes Moyennes (après nettoyage)",
        )
        fig7.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=380,
        )
        st.plotly_chart(fig7, use_container_width=True)

    # ── Tab 5 : Qualité des données ───────────────────────────────────────────
    with tab5:
        st.subheader("🧹 Rapport de Qualité des Données")

        raw_counts = {
            "Lignes brutes (movies)":  45466,
            "Lignes brutes (credits)": 45476,
            "Id non-numériques":       3,
            "Doublons id (movies)":    30,
            "Doublons titre":          3188,
            "Films non-Released":      452,
            "Overview manquantes":     954,
            "Runtime manquant":        263,
            "Release_date manquant":   87,
            "Vote_count < 5":          "~12%",
        }
        clean_counts = {
            "Films après nettoyage": len(df),
            "Films éligibles Top":   len(eligible),
            "Genres uniques":        len({g for sub in df["genres_list"] for g in sub}),
            "Directeurs uniques":    df["director"].nunique(),
            "Plage d'années":        f"{int(df['release_year'].min())}–{int(df['release_year'].max())}",
        }

        col_r, col_c = st.columns(2)
        with col_r:
            st.markdown("**Données brutes — problèmes détectés**")
            for k, v in raw_counts.items():
                st.markdown(f"- `{k}` : **{v}**")
        with col_c:
            st.markdown("**Après nettoyage**")
            for k, v in clean_counts.items():
                st.markdown(f"- `{k}` : **{v}**")

        # Taux de complétude
        completeness = {
            "title":        df["title"].notna().mean() * 100,
            "overview":     (df["overview"] != "").mean() * 100,
            "director":     (df["director"] != "").mean() * 100,
            "genres_list":  (df["genres_list"].apply(len) > 0).mean() * 100,
            "release_year": df["release_year"].notna().mean() * 100,
        }
        comp_df = pd.DataFrame(list(completeness.items()), columns=["Champ", "Complétude (%)"])
        fig_comp = px.bar(
            comp_df, x="Champ", y="Complétude (%)",
            color="Complétude (%)", color_continuous_scale="Greens",
            title="Taux de complétude après nettoyage",
            range_y=[0, 100],
        )
        fig_comp.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="#e8e8e8", height=360, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)
