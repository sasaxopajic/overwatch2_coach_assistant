import streamlit as st
import ow2_langchain as lch

# Apply Poppins font to the whole app
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
    html * {
        font-family: 'Poppins', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("./assets/ow2_logo.png", use_column_width=True)
st.markdown("<h1 style='text-align: center'>Type: 3C-HO - Overwatch 2 Assistant</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center'>All strategies you want to know!</h4>", unsafe_allow_html=True)

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
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.write(chat["content"])
            else:
                # Display the assistant's image and then the message content
                with st.chat_message("assistant", avatar="./assets/ow2_icon.png"):
                    st.markdown("<h4 style='padding: 0; margin-bottom: 1.5rem; font-weight: 900; color: #F99E1A'>Type: 3C-HO</h4>", unsafe_allow_html=True)
                    st.write(chat["content"],)
