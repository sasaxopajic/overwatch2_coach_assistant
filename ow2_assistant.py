import streamlit as st
import ow2_langchain as lch

st.header("Overwatch 2 assistant")
st.subheader("Ask any question about strategies that you want to know!")

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="Enter the URL of the video:",
            max_chars=50
            )
        query = st.text_area(
            label="How can I help you?",
            key="query"
            )
        submit_button = st.form_submit_button(label='Submit')


# Process user input and generate response
if submit_button and query and youtube_url:
    # Create the database and get the response
    db = lch.load_or_create_vector_database_from_yt_url(youtube_url)
    response = lch.get_response_from_query(db, query)

    # Store the user query and response in chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])
