import streamlit as st
import ow2_langchain as lch

st.header("Overwatch 2 assistant")
st.subheader("Ask any question about strategies that you want to know!")

import textwrap

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="What is the YouTube video URL?",
            max_chars=50
            )
        query = st.text_area(
            label="Ask me about the video?",
            key="query"
            )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url:
        db = lch.create_vector_database_from_yt_url(youtube_url)
        response = lch.get_response_from_query(db, query)
        st.subheader("Answer:")
        st.markdown(response)