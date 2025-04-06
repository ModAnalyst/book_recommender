# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import networkx as nx
import pickle

# ----------------------------------------
# Page Config
# ----------------------------------------
st.set_page_config(page_title="📚 Hybrid Book Recommender", layout="wide")

# ----------------------------------------
# Styled Header
# ----------------------------------------
st.markdown("""
<div style='text-align: center; padding: 10px 0 0 0;'>
    <h1 style='color:#ff6347;'>📚 Hybrid Book Recommender</h1>
    <p style='font-size: 18px;'>Get personalized book suggestions using Content, Collaborative, and Knowledge Graph Filtering.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------
# Sidebar Style & Inputs
# ----------------------------------------
st.markdown("""
    <style>
    .st-emotion-cache-1lcbmhc {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔍 Search for a Book")
    search_option = st.radio("Search by:", ["Title", "Author", "Publisher"])
    search_query = st.text_input(f"Enter part of the {search_option.lower()}")

# ----------------------------------------
# Load Data & Models
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/books.csv', on_bad_lines='skip')
    df.fillna({'authors': '', 'publisher': '', 'language_code': '', 'title': '', 'average_rating': 0}, inplace=True)
    if 'user_id' not in df.columns:
        df['user_id'] = pd.Series(range(1, len(df) + 1))
    df['user_id'] = df['user_id'].astype(int)
    return df

df = load_data()
tfidf_matrix = joblib.load('/content/tfidf_matrix.pkl')
cbf_sim_df = joblib.load('/content/cbf_sim_df.pkl')
user_book_matrix = joblib.load('/content/user_book_matrix.pkl')
user_similarity = joblib.load('/content/user_similarity.pkl')
G = pickle.load(open('/content/knowledge_graph.pickle', 'rb'))

# ----------------------------------------
# Recommendation Functions
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
# Run Recommendation
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
        book_title = match.iloc[0]['title']
        user_id = 1  # Default user
        st.markdown(f"✅ **Recommendations based on:** _{book_title}_")

        with st.spinner("Generating recommendations..."):
            recs = hybrid_recommend(user_id, book_id, top_n=5)

        if not recs.empty:
            st.subheader("📚 Recommended Books")
            st.dataframe(recs.set_index('bookID'), use_container_width=True)
        else:
            st.warning("No recommendations found.")
    else:
        st.warning("❌ No matching books found.")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("""
<hr style='border: 1px solid #eee;' />
<p style='text-align: center; font-size: 14px; color: gray;'>
  🚀 Built with ❤️ using Streamlit · Hybrid Recommendation System · 2024
</p>
""", unsafe_allow_html=True)
