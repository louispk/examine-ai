import streamlit as st
import openai
import os
import json
from datetime import datetime
from typing import List, Dict

# Make sure to set the OPENAI_API_KEY in your environment variables.
openai.api_key = os.getenv("OPENAI_API_KEY")


class MessageStorage:
    """
    A class for storing chat messages with a timestamp to a file.
    """

    def __init__(self, file_path: str = "chat_history.json"):
        self._file_path = file_path
        self._initialize_messages()

    def _initialize_messages(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def store_message(self, message: Dict[str, str]):
        """Stores a message dictionary with a timestamp to a JSON file."""
        message_with_timestamp = {"timestamp": datetime.now().isoformat(), **message}
        with open(self._file_path, "a") as f:
            json.dump(message_with_timestamp, f)
            f.write("\n")


class OpenAIResponder:
    """
    A class to handle responses from OpenAI's GPT model.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Fetches a response from OpenAI using the given list of message dictionaries."""
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
    """
    A class containing static methods for UI interactions.
    """

    @staticmethod
    def display_messages(column, messages: List[Dict[str, str]]):
        """Displays messages in a Streamlit chat."""
        for message in messages:
            with column:
                role = message["role"]
                with st.container():
                    if role == "user":
                        st.write(f"You: {message['content']}")
                    else:
                        st.info(f"AI: {message['content']}")

    @staticmethod
    def get_user_input(column, key: str, callback):
        return column.text_input(
            "Talk to the chatbot",
            key=key,
            on_change=callback,
            args=(key,),
        )

    def handle_input_change(column_key: str):
        user_input = st.session_state[column_key]
        if user_input:  # Proceed only if the user has entered text
            # Append the user message to the conversation
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Clear the input box after the message is sent
            st.session_state[column_key] = ""

            # Fetch and store the response
            responder = OpenAIResponder(api_key=openai.api_key)
            response = responder.get_response(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state["last_ai_response"] = response

            # Store the messages using MessageStorage
            message_storage = MessageStorage()
            message_storage.store_message({"role": "user", "content": user_input})
            message_storage.store_message({"role": "assistant", "content": response})


class SafeguardAI:
    """
    A class to evaluate and display AI response safety based on predefined principles.
    """

    def __init__(
        self,
        responder: OpenAIResponder,
        principles_file_path: str = "core_principles.json",
        scores_file_path: str = "safety_scores.json",
    ):
        self._responder = responder
        self._principles_file_path = principles_file_path
        self._scores_file_path = scores_file_path
        self._safety_principles = self._load_safety_principles()

    def _load_safety_principles(self) -> List[str]:
        """Loads safety principles from a JSON file."""
        with open(self._principles_file_path, "r") as file:
            principles = json.load(file)
        return principles.get("principles", [])

    def _evaluate_principle(self, response: str, principle: str) -> int:
        """Evaluates a single safety principle."""
        prompt = f"Please evaluate the following response on a scale from 0 to 10 based on the principle: '{principle}'.\nResponse: \"{response}\""
        model_response = self._responder.get_response(
            [{"role": "system", "content": prompt}]
        )
        try:
            score = int(model_response.strip())
            score = max(0, min(10, score))
        except ValueError:
            score = 99
        return score

    def get_safety_scores(self, response: str) -> Dict[str, int]:
        """Evaluates safety principles for a given response."""
        return {
            principle["name"]: self._evaluate_principle(response, principle)
            for principle in self._safety_principles
        }

    def save_safety_scores(self, scores: Dict[str, int]):
        """Saves the safety scores to a JSON file."""
        with open(self._scores_file_path, "w") as file:
            json.dump(scores, file, indent=4)

    def display_safeguard_evaluation(self, column, response: str):
        """Displays and saves the safety evaluation for a response."""
        safety_scores = self.get_safety_scores(response)

        # Calculate the overall score
        overall_score = sum(safety_scores.values()) / len(safety_scores)

        with column:
            # Construct message with individual scores and the overall score
            scores_message = "<br>".join(
                f"Principle: {principle}, Score: {score}"
                for principle, score in safety_scores.items()
            )
            scores_message += f"<br><br><strong>Overall Score: {overall_score:.2f}</strong>"  # Strong tag to emphasize the overall score

            # Display the message with green bubble styling
            success_html = f"""
            <div style="
                border-radius: 4px;
                padding: 10px;
                color: #FFF;
                background-color: #28a745;
            ">
                {scores_message}
            </div>
            """
            st.markdown(success_html, unsafe_allow_html=True)

        # Save the safety scores
        self.save_safety_scores(safety_scores)


def main():
    st.title("examine|AI")
    st.subheader("Your assistant for trustworthy conversations with AI")
    st.image("logo.png", width=200)

    # Initialize session state
    MessageStorage()  # Ensures messages are initialized in session state
    if "last_ai_response" not in st.session_state:
        st.session_state.last_ai_response = None
    if "evaluate_safeguard" not in st.session_state:
        st.session_state.evaluate_safeguard = False

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Primary AI")
        ChatUI.display_messages(col1, st.session_state.messages)
        ChatUI.get_user_input(col1, "user_input_primary_ai", ChatUI.handle_input_change)

    # Safeguard AI column
    with col2:
        st.subheader("Safeguard AI")
        if st.session_state.last_ai_response:
            if col2.button("Evaluate", key="evaluate"):
                st.session_state.evaluate_safeguard = True
                st.experimental_rerun()

        if st.session_state.evaluate_safeguard:
            # st.write("test")  # You should see this when 'Evaluate' is clicked.
            responder = OpenAIResponder(api_key=openai.api_key)
            safeguard_ai = SafeguardAI(responder)
            safeguard_ai.display_safeguard_evaluation(
                col2, st.session_state.last_ai_response
            )


if __name__ == "__main__":
    main()
