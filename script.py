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

