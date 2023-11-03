import streamlit as st
import openai
import os
import json
from datetime import datetime

class MessageStorage:
    def __init__(self, file_path: str = 'chat_history.json'):
        self.file_path = file_path
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def store_message(self, message: dict):
        message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(message_with_timestamp) + '\n')

class OpenAIResponder:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_response(self, messages: list) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message['content']

class ChatUI:
    @staticmethod
    def display_messages(column, messages):
        for message in messages:
            with column:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    @staticmethod
    def get_user_input(column):
        return column.text_input("Talk to the chatbot", key=f"user_input_{column}")

    @staticmethod
    def send_button_pressed(column):
        return column.button('Send', key=f"send_{column}")

    @staticmethod
    def display_safety_evaluation(column, safeguard_ai, response):
        safety_scores = safeguard_ai.get_safety_scores(response)
        for principle, score in safety_scores.items():
            column.info(f"Principle: {principle}, Score: {score}")

class SafeguardAI:
    def __init__(self, responder, principles_file_path: str = 'core_principles.json', scores_file_path: str = 'safety_scores.json'):
        self.responder = responder
        self.principles_file_path = principles_file_path
        self.scores_file_path = scores_file_path
        self.safety_principles = self.load_safety_principles()

    def load_safety_principles(self):
        with open(self.principles_file_path, 'r') as file:
            principles = json.load(file)
        return principles.get("principles", [])

    def evaluate_principle(self, response: str, principle: str) -> int:
        prompt = f"Please evaluate the following response on a scale from 0 to 10 based on the principle: '{principle}'.\nResponse: \"{response}\""
        model_response = self.responder.get_response([{"role": "system", "content": prompt}])
        try:
            score = int(model_response.strip())
            score = max(0, min(10, score))
        except ValueError:
            score = 99
        return score
    
    def get_safety_scores(self, response: str) -> dict:
        safety_scores = {}
        for principle in self.safety_principles:
            principle_name = principle['name']
            score = self.evaluate_principle(response, principle)
            safety_scores[principle_name] = score
        return safety_scores

    def save_safety_scores(self, scores: dict):
        with open(self.scores_file_path, 'w') as file:
            json.dump(scores, file, indent=4)

    def display_safeguard_evaluation(self, column, response):
        safety_scores = self.get_safety_scores(response)
        with column:
            for principle, score in safety_scores.items():
                st.info(f"Principle: {principle}, Score: {score}")
            self.save_safety_scores(safety_scores)


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.title('examine|AI')
    st.subheader("Your assistant for trustworthy conversations with AI")
    st.image('logo.png', width=200)
    col1, col2 = st.columns(2)

    with col1:
        
        st.subheader("Primary AI")

    with col2:
        st.subheader("Safeguard AI")

    message_storage = MessageStorage()
    responder = OpenAIResponder(api_key=openai.api_key)
    safeguard_ai = SafeguardAI(responder)

    # Primary AI interaction
    ChatUI.display_messages(col1, st.session_state.messages)
    user_input = ChatUI.get_user_input(col1)
    if user_input and ChatUI.send_button_pressed(col1):
        st.session_state.messages.append({"role": "user", "content": user_input})
        message_storage.store_message({"role": "user", "content": user_input})
        response = responder.get_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message_storage.store_message({"role": "assistant", "content": response})
        ChatUI.display_messages(col1, [{"role": "assistant", "content": response}])

        # Safeguard AI evaluation
        safeguard_ai.display_safeguard_evaluation(col2, response)

    # Interaction with Safeguard AI based on output evaluation
    with col2:
        safeguard_user_input = st.text_input("Talk to the Safeguard AI", key="safeguard_input")
        if safeguard_user_input and st.button("Evaluate", key="evaluate"):
            safeguard_ai.display_safeguard_evaluation(col2, safeguard_user_input)
            st.session_state.messages.append({"role": "safeguard_user", "content": safeguard_user_input})
            message_storage.store_message({"role": "safeguard_user", "content": safeguard_user_input})

if __name__ == "__main__":
    main()

# import streamlit as st
# import openai
# import os
# import json
# from datetime import datetime

# class MessageStorage:
#     def __init__(self, file_path: str = 'chat_history.json'):
#         self.file_path = file_path
#         if "messages" not in st.session_state:
#             st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

#     def store_message(self, message: dict):
#         message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
#         with open(self.file_path, 'a') as f:
#             f.write(json.dumps(message_with_timestamp) + '\n')

#     def read_chat_history(self):
#         with open(self.file_path, 'r') as f:
#             for line in f:
#                 print(json.loads(line))

# class OpenAIResponder:
#     def __init__(self, api_key: str):
#         self.api_key = api_key

#     def get_response(self, messages: list) -> str:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages
#         )
#         return response.choices[0].message['content']

# class ChatUI:
#     @staticmethod
#     def display_messages():
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#     @staticmethod
#     def get_user_input():
#         return st.text_input("Talk to the chatbot")

#     @staticmethod
#     def send_button_pressed():
#         return st.button('Send')
    
#     @staticmethod
#     def display_safety_evaluation(safeguard_ai, response):
#         # Evaluate the response using SafeguardAI
#         safety_scores = safeguard_ai.get_safety_scores(response)
#         # Display the safety scores
#         for principle, score in safety_scores.items():
#             st.info(f"Principle: {principle}, Score: {score}")
#         # Save the scores
#         safeguard_ai.save_safety_scores(safety_scores)

# class SafeguardAI:
#     def __init__(self, responder, principles_file_path: str = 'core_principles.json', scores_file_path: str = 'safety_scores.json'):
#         self.responder = responder
#         self.principles_file_path = principles_file_path
#         self.scores_file_path = scores_file_path
#         self.safety_principles = self.load_safety_principles()

#     def load_safety_principles(self):
#         with open(self.principles_file_path, 'r') as file:
#             principles = json.load(file)
#         return principles.get("principles", [])

#     def evaluate_principle(self, response: str, principle: str) -> int:
#         # Call the language model to evaluate the response against the principle.
#         # The model should return a score from 0 to 10.
#         prompt = f"Please evaluate the following response on a scale from 0 to 10 based on the principle: '{principle}'.\nResponse: \"{response}\""
#         model_response = self.responder.get_response([{"role": "system", "content": prompt}])
#         # Parse the model's response to extract the score.
#         try:
#             # Assuming the model returns a plain number as a response.
#             score = int(model_response.strip())
#             # Ensure the score is within the expected range.
#             score = max(0, min(10, score))
#         except ValueError:
#             # In case the model doesn't return a valid integer.
#             score = 99
#         return score
    
#     def get_safety_scores(self, response: str) -> dict:
#         safety_scores = {}
#         for principle in self.safety_principles:
#             principle_name = principle['name']
#             score = self.evaluate_principle(response, principle)
#             safety_scores[principle_name] = score
#         self.save_safety_scores(safety_scores)
#         return safety_scores

#     def save_safety_scores(self, scores: dict):
#         with open(self.scores_file_path, 'w') as file:
#             json.dump(scores, file, indent=4)

# def main():
#     # Set up
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     st.title('examine|AI')
#     st.subheader("Your assistant for trustworthy conversations with AI")
#     st.image('logo.png', width=200)

#     # Initialize objects
#     message_storage = MessageStorage()
#     responder = OpenAIResponder(api_key=openai.api_key)

#     # Initialize SafeguardAI
#     safeguard_ai = SafeguardAI(responder)

#     # Display previous messages
#     ChatUI.display_messages()

#     # Handle user input
#     user_input = ChatUI.get_user_input()
#     if user_input and ChatUI.send_button_pressed():
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         message_storage.store_message({"role": "user", "content": user_input})

#         # Get and display response
#         response = responder.get_response(st.session_state.messages)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         message_storage.store_message({"role": "assistant", "content": response})

#         with st.chat_message("assistant"):
#             st.markdown(f'Bot: {response}')

#         # Check the response with SafeguardAI
#         ChatUI.display_safety_evaluation(safeguard_ai, response)

#         # Clear the input box
#         st.text_input("Talk to the Safeguard AI", value='')

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# import openai
# import os
# import json
# from datetime import datetime

# # Separating OpenAI chat logic into a class following the Single Responsibility Principle
# class OpenAIChatbot:
#     def __init__(self, api_key, storage_file='chat_history.json'):
#         self.api_key = api_key
#         self.storage_file = storage_file
#         self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

#     def send_message(self, message):
#         user_message = {"role": "user", "content": message_text}
#         self.messages.append(user_message)
#         self.store_message(user_message)
#         return self.get_response()

#     def get_response(self):
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=self.messages
#         )
#         assistant_message = response.choices[0].message
#         self.messages.append(assistant_message)
#         self.store_message(assistant_message)
#         return assistant_message['content']

#     def store_message(self, message):
#         # Add timestamp to message
#         message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
#         # Append the message with timestamp to the storage file
#         with open(self.storage_file, 'a') as f:
#             f.write(json.dumps(message_with_timestamp) + '\n')

#     def read_chat_history(file_path='chat_history.json'):
#         with open(file_path, 'r') as f:
#             for line in f:
#                 print(json.loads(line))

# # Instantiating the chatbot
# openai.api_key = os.getenv("OPENAI_API_KEY")

# # Streamlit app setup
# st.title('examine|AI')
# st.subheader("your assistant for trustworthy conversations with AI")
# st.image('logo.png', width=200)

# # User text input
# user_input = st.text_input("Talk to the chatbot")

# # Use Streamlit's session state to keep the chatbot instance
# if 'chatbot' not in st.session_state:
#     st.session_state.chatbot = OpenAIChatbot(api_key=openai.api_key)

# # Handle button press
# if st.button('Send'):
#     if user_input:
#         # Send the user's message and get the response from OpenAI
#         response = st.session_state.chatbot.send_message(user_input)
#         st.write(f'Bot: {response}')
#         # Clear the input box after sending the message
#         st.session_state.user_input = ''
#     else:
#         st.write('Please enter some text to chat.')

# # To run the app, save this script as `app.py` and run the command `streamlit run app.py`


# # # Title of the web app
# # st.title('My First Streamlit App')

# # # Text input for the user's name
# # name = st.text_input('Enter your name')

# # # Button to display the greeting
# # if st.button('Say hello'):
# #     # Display the greeting
# #     st.write(f'Hello {name}!')

# # # To run the app, save this script as `app.py` and run the command `streamlit run app.py`

