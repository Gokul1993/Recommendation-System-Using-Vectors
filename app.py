import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(layout="wide")
st.title("🛒 Recommendation System (Compact UI)")

# -----------------------------
# PRODUCTS
# -----------------------------
products = {
    "P1": {"name": "Electronics", "vector": [1,0,0,0], "img": "images/electronics.jpg"},
    "P2": {"name": "Books",       "vector": [0,0,0,1], "img": "images/books.jpg"},
    "P3": {"name": "Sports",      "vector": [0,1,0,0], "img": "images/sports.jpg"},
    "P4": {"name": "Entertainment","vector": [0,0,1,0], "img": "images/entertainment.jpg"},
}

categories = ["Electronics", "Sports", "Entertainment", "Books"]

# -----------------------------
# SESSION INIT
# -----------------------------
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

users = ["U1", "U2"]
user = st.selectbox("Select User", users)

if user not in st.session_state.user_data:
    st.session_state.user_data[user] = {
        "click": np.zeros(4),
        "cart": np.zeros(4),
        "purchase": np.zeros(4)
    }

session = st.session_state.user_data[user]

# -----------------------------
# TOP SECTION (Products)
# -----------------------------
st.write("## 🛍️ Products")
cols = st.columns(4)

for i, (pid, pdata) in enumerate(products.items()):
    with cols[i]:
        st.subheader(pid)
        st.caption(pdata["name"])

        if os.path.exists(pdata["img"]):
            st.image(pdata["img"], use_container_width=True)
        else:
            st.warning("Image missing")

        vec = np.array(pdata["vector"])

        c1, c2, c3 = st.columns(3)

        if c1.button("Click", key=f"c_{pid}"):
            session["click"] += vec

        if c2.button("Cart", key=f"ca_{pid}"):
            session["cart"] += vec

        if c3.button("Buy", key=f"b_{pid}"):
            session["purchase"] += vec

# -----------------------------
# SINGLE ROW: ACTIVITY + SUMMARY + VECTOR
# -----------------------------
colA, colB, colC = st.columns([2,1,1])

# ---- Activity Table (LEFT)
with colA:
    st.write("### 📊 Activity")

    df = pd.DataFrame({
        "Category": categories,
        "Clicks": session["click"],
        "Cart": session["cart"],
        "Buy": session["purchase"]
    })

    st.dataframe(df, use_container_width=True, height=220)

# ---- Summary (CENTER)
with colB:
    st.write("### 🔍 Summary")

    total_clicks = int(np.sum(session["click"]))
    total_cart = int(np.sum(session["cart"]))
    total_buy = int(np.sum(session["purchase"]))

    st.metric("Clicks", total_clicks)
    st.metric("Cart", total_cart)
    st.metric("Buy", total_buy)

    activity_score = session["click"] + 2*session["cart"] + 3*session["purchase"]

    if np.sum(activity_score) > 0:
        fav = categories[np.argmax(activity_score)]
        st.success(f"⭐ {fav}")
    else:
        st.info("No activity")

# ---- User Vector (RIGHT)
with colC:
    st.write("### 🧠 Vector")

    def get_user_vector(s):
        act = s["click"] + 2*s["cart"] + 3*s["purchase"]
        if np.max(act) == 0:
            return np.zeros(4)
        return act / np.max(act)

    user_vector = get_user_vector(session)

    st.write(np.round(user_vector, 2))

# -----------------------------
# RECOMMENDATION (BOTTOM)
# -----------------------------
if st.button("🔍 Recommend"):

    st.write("## 🎯 Recommendation")

    scores = []

    for pid, pdata in products.items():
        sim = cosine_similarity([user_vector], [pdata["vector"]])[0][0]
        scores.append((pid, sim))

    scores.sort(key=lambda x: x[1], reverse=True)

    # Show horizontally
    rec_cols = st.columns(4)

    for i, (pid, score) in enumerate(scores):
        with rec_cols[i]:
            st.write(f"**{pid}**")
            st.write(f"Score: {round(score,2)}")

    st.success(f"Top: {scores[0][0]}")