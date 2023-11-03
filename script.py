import streamlit as st
import openai
import os
import json
from datetime import datetime


class MessageStorage:
    def __init__(self, file_path: str = "chat_history.json"):
        self.file_path = file_path
        # Initialize the messages in the session state if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def store_message(self, message: dict):
        message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
        with open(self.file_path, "a") as f:
            f.write(json.dumps(message_with_timestamp) + "\n")


class OpenAIResponder:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_response(self, messages: list) -> str:
        # Filter out any messages that do not have a role recognized by the OpenAI API
        valid_messages = [
            msg
            for msg in messages
            if msg["role"] in ["system", "assistant", "user", "function"]
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=valid_messages
        )
        return response.choices[0].message["content"]


class ChatUI:
    @staticmethod
    def display_messages(column, messages):
        for message in messages:
            with column:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    @staticmethod
    def get_user_input(column):
        return column.text_input(
            "Talk to the chatbot",
            key=f"user_input_{column}",
            on_change=ChatUI.handle_input_change,
            args=(column,),
        )

    def handle_input_change():
        user_input_key = "user_input_primary_ai"
        user_input = st.session_state[user_input_key]
        if user_input:  # Proceed only if the user has entered text
            # Append the user message to the conversation
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Clear the input box after the message is sent
            st.session_state[user_input_key] = ""

            # Fetch and store the response
            responder = OpenAIResponder(api_key=openai.api_key)
            response = responder.get_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message_storage = MessageStorage()
            message_storage.store_message({"role": "user", "content": user_input})
            message_storage.store_message({"role": "assistant", "content": response})

            # We will not call display_messages here because Streamlit will rerun the whole script
            # after this and the display_messages call in the main function will do the job.

    @staticmethod
    def display_safety_evaluation(column, safeguard_ai, response):
        safety_scores = safeguard_ai.get_safety_scores(response)
        for principle, score in safety_scores.items():
            column.info(f"Principle: {principle}, Score: {score}")


class SafeguardAI:
    def __init__(
        self,
        responder,
        principles_file_path: str = "core_principles.json",
        scores_file_path: str = "safety_scores.json",
    ):
        self.responder = responder
        self.principles_file_path = principles_file_path
        self.scores_file_path = scores_file_path
        self.safety_principles = self.load_safety_principles()

    def load_safety_principles(self):
        with open(self.principles_file_path, "r") as file:
            principles = json.load(file)
        return principles.get("principles", [])

    def evaluate_principle(self, response: str, principle: str) -> int:
        prompt = f"Please evaluate the following response on a scale from 0 to 10 based on the principle: '{principle}'.\nResponse: \"{response}\""
        # st.write(prompt)
        model_response = self.responder.get_response(
            [{"role": "system", "content": prompt}]
        )
        try:
            score = int(model_response.strip())
            score = max(0, min(10, score))
        except ValueError:
            score = 99
        return score

    def get_safety_scores(self, response: str) -> dict:
        safety_scores = {}
        for principle in self.safety_principles:
            principle_name = principle["name"]
            score = self.evaluate_principle(response, principle)
            safety_scores[principle_name] = score
        return safety_scores

    def save_safety_scores(self, scores: dict):
        with open(self.scores_file_path, "w") as file:
            json.dump(scores, file, indent=4)

    def display_safeguard_evaluation(self, column, response):
        safety_scores = self.get_safety_scores(response)
        with column:
            for principle, score in safety_scores.items():
                st.info(f"Principle: {principle}, Score: {score}")
            self.save_safety_scores(safety_scores)


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.title("examine|AI")
    st.subheader("Your assistant for trustworthy conversations with AI")
    st.image("logo.png", width=200)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_ai_response" not in st.session_state:
        st.session_state.last_ai_response = None
    if "evaluate_safeguard" not in st.session_state:
        st.session_state.evaluate_safeguard = False

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primary AI")
        ChatUI.display_messages(col1, st.session_state.messages)
        user_input = col1.text_input("Talk to the chatbot", key="user_input_primary_ai")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            responder = OpenAIResponder(api_key=openai.api_key)
            response = responder.get_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.last_ai_response = response
            st.write(st.session_state.last_ai_response)
            # Clear the input
            # st.session_state['user_input_primary_ai'] = ''

    with col2:
        st.subheader("Safeguard AI")
        if st.session_state.last_ai_response:
            st.write("Latest AI response to evaluate:")
            st.write(st.session_state.last_ai_response)
            if col2.button("Evaluate", key="evaluate"):
                st.session_state.evaluate_safeguard = True

        if st.session_state.evaluate_safeguard:
            safeguard_ai = SafeguardAI(responder)
            safeguard_ai.display_safeguard_evaluation(
                col2, st.session_state.last_ai_response
            )
            safeguard_query = col2.text_input(
                "Ask the Safeguard AI", key="safeguard_query"
            )
            if safeguard_query:
                # Handle the safeguard query and display the response.
                safeguard_response = safeguard_ai.get_safety_scores(safeguard_query)
                st.write(
                    safeguard_response
                )  # This will display the safeguard AI's response


if __name__ == "__main__":
    main()
