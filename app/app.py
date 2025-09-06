import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from dotenv import load_dotenv
from anime_pipeline.pipeline import AnimeRecommendationPipeline
st.set_page_config(page_title="Anime Recommender", layout="wide")


load_dotenv()

@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()

pipeline = init_pipeline()

st.title("Anime Recommender System")

query = st.text_input("Enter your anime preferences e.g.: light-hearted anime with school settings")
if query:
    with st.spinner("Fetching recommendations for you..."):
        response = pipeline.recommend(query)
        st.markdown("### Recommendations")
        st.write(response)
