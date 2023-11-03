import streamlit as st
import pandas as pd
import openai

# Title of the web app
st.title('My First Streamlit App')

# Text input for the user's name
name = st.text_input('Enter your name')

# Button to display the greeting
if st.button('Say hello'):
    # Display the greeting
    st.write(f'Hello {name}!')

# To run the app, save this script as `app.py` and run the command `streamlit run app.py`

