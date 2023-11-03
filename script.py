import streamlit as st
import pandas as pd
import openai
import os
import json
from datetime import datetime

# Separating OpenAI chat logic into a class following the Single Responsibility Principle
class OpenAIChatbot:
    def __init__(self, api_key, storage_file='chat_history.json'):
        self.api_key = api_key
        self.storage_file = storage_file
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def send_message(self, message):
        user_message = {"role": "user", "content": message_text}
        self.messages.append(user_message)
        self.store_message(user_message)
        return self.get_response()

    def get_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        assistant_message = response.choices[0].message
        self.messages.append(assistant_message)
        self.store_message(assistant_message)
        return assistant_message['content']

    def store_message(self, message):
        # Add timestamp to message
        message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
        # Append the message with timestamp to the storage file
        with open(self.storage_file, 'a') as f:
            f.write(json.dumps(message_with_timestamp) + '\n')

    def read_chat_history(file_path='chat_history.json'):
        with open(file_path, 'r') as f:
            for line in f:
                print(json.loads(line))

# Instantiating the chatbot
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app setup
st.title('examine|AI')
st.subheader("your assistant for trustworthy conversations with AI")
st.image('logo.png', width=200)

# User text input
user_input = st.text_input("Talk to the chatbot")

# Use Streamlit's session state to keep the chatbot instance
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = OpenAIChatbot(api_key=openai.api_key)

# Handle button press
if st.button('Send'):
    if user_input:
        # Send the user's message and get the response from OpenAI
        response = st.session_state.chatbot.send_message(user_input)
        st.write(f'Bot: {response}')
        # Clear the input box after sending the message
        st.session_state.user_input = ''
    else:
        st.write('Please enter some text to chat.')

# To run the app, save this script as `app.py` and run the command `streamlit run app.py`


# # Title of the web app
# st.title('My First Streamlit App')

# # Text input for the user's name
# name = st.text_input('Enter your name')

# # Button to display the greeting
# if st.button('Say hello'):
#     # Display the greeting
#     st.write(f'Hello {name}!')

# # To run the app, save this script as `app.py` and run the command `streamlit run app.py`

