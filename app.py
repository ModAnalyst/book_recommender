import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import pickle
import gzip
import os
import gdown

# ----------------------------------------
# 🎯 Streamlit Page Config
# ----------------------------------------
st.set_page_config(page_title="📚 Hybrid Book Recommender", layout="wide")
st.title("📚 Hybrid Book Recommender System")
st.markdown("Combines Content-Based, Collaborative, and Knowledge Graph filtering.")

# ----------------------------------------
# 📦 Load Data
# ----------------------------------------
@st.cache_resource
@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", on_bad_lines='skip')
    df['authors'] = df['authors'].fillna('')
    df['publisher'] = df['publisher'].fillna('')
    df['language_code'] = df['language_code'].fillna('')
    df['title'] = df['title'].fillna('')
    df['average_rating'] = df['average_rating'].fillna(0)

    # Ensure user_id column exists
    if 'user_id' not in df.columns:
        df['user_id'] = pd.Series(range(1, len(df) + 1))
    df['user_id'] = df['user_id'].astype(int)
    return df

# ----------------------------------------
# 📥 Download CBF model from Google Drive
# ----------------------------------------
@st.cache_resource
def load_cbf_model():
    file_id = "1oafJ95wpLUiyBMH4eDuSH9IQlyP3p4wx"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "cbf_sim_df.pkl.gz"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    with gzip.open(output, "rb") as f:
        return joblib.load(f)

# ----------------------------------------
# 💾 Load Models
# ----------------------------------------
@st.cache_resource
def load_models():
    return {
        "tfidf_matrix": joblib.load("tfidf_matrix.pkl"),
        "cbf_sim_df": load_cbf_model(),
        "user_book_matrix": joblib.load("user_book_matrix.pkl"),
        "user_similarity": joblib.load("user_similarity.pkl"),
        "kg_graph": pickle.load(open("knowledge_graph.pickle", "rb"))
    }

models = load_models()
tfidf_matrix = models["tfidf_matrix"]
cbf_sim_df = models["cbf_sim_df"]
user_book_matrix = models["user_book_matrix"]
user_similarity = models["user_similarity"]
G = models["kg_graph"]

# ----------------------------------------
# 🔍 Recommendation Functions
# ----------------------------------------
def recommend_cbf(book_id, top_n=5):
    if book_id not in cbf_sim_df.index:
        return []
    scores = cbf_sim_df[book_id].sort_values(ascending=False)[1:top_n+1]
    return scores.index.tolist()

def recommend_cf(user_id, top_n=5):
    if user_id not in user_similarity.index:
        return []
    similar_users = user_similarity[user_id].drop(user_id).nlargest(5)
    weighted_scores = user_book_matrix.loc[similar_users.index].T.dot(similar_users)
    normalized_scores = weighted_scores / (similar_users.sum() + 1e-9)
    user_books = user_book_matrix.loc[user_id]
    unseen_books = normalized_scores[user_books == 0]
    return unseen_books.sort_values(ascending=False).head(top_n).index.tolist()

def recommend_kg(book_id, top_n=5):
    node_id = f"Book_{book_id}"
    if node_id not in G:
        return []
    neighbors = set(G.neighbors(node_id))
    similar_books = set()
    for node in neighbors:
        if G.nodes[node].get('label') in ['Author', 'Publisher', 'Language']:
            for nb in G.neighbors(node):
                if nb.startswith("Book_") and nb != node_id:
                    similar_books.add(nb.replace("Book_", ""))
    return list(map(int, list(similar_books)))[:top_n]

def hybrid_recommend(user_id, book_id, top_n=5):
    cbf_recs = recommend_cbf(book_id, top_n=top_n*2)
    cf_recs = recommend_cf(user_id, top_n=top_n*2)
    kg_recs = recommend_kg(book_id, top_n=top_n*2)
    all_recs = cbf_recs + cf_recs + kg_recs
    rec_scores = pd.Series(all_recs).value_counts()
    top_recs = rec_scores.sort_values(ascending=False).head(top_n).index
    titles = df[df['bookID'].isin(top_recs)][['bookID', 'title', 'authors', 'publisher']].drop_duplicates()
    return titles

# ----------------------------------------
# 🎛️ Sidebar Search Interface
# ----------------------------------------
st.sidebar.header("🔍 Search for a Book")
search_option = st.sidebar.radio("Search by:", ["Title", "Author", "Publisher"])
search_query = st.sidebar.text_input(f"Enter part of the {search_option.lower()}")

# ----------------------------------------
# 🚀 Run Recommendation
# ----------------------------------------
if st.sidebar.button("📖 Recommend Books") and search_query:
    if search_option == "Title":
        match = df[df['title'].str.contains(search_query, case=False, na=False)]
    elif search_option == "Author":
        match = df[df['authors'].str.contains(search_query, case=False, na=False)]
    else:
        match = df[df['publisher'].str.contains(search_query, case=False, na=False)]

    if not match.empty:
        book_id = match.iloc[0]['bookID']
        user_id = 1
        with st.spinner("Generating recommendations..."):
            recs = hybrid_recommend(user_id, book_id, top_n=5)

        if not recs.empty:
            st.subheader("📚 Recommended Books")
            st.table(recs.set_index('bookID'))
        else:
            st.warning("⚠️ No recommendations found.")
    else:
        st.warning(f"❌ No books found for '{search_query}'. Try another keyword.")
