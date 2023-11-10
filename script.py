import streamlit as st
import openai
import os
import json
from datetime import datetime
from typing import List, Dict
import re
import random
from utils import extract_score_from_evaluation, get_color, text_to_html
from prompts import safeguard_assessment

# Make sure to set the OPENAI_API_KEY in your environment variables.
openai.api_key = os.getenv('OPENAI_API_KEY')


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
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self._api_key = api_key
        self._model = model

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Fetches a response from OpenAI using the given list of message dictionaries."""
        valid_messages = [
            msg
            for msg in messages
            if msg["role"] in ["system", "assistant", "user", "function"]
        ]
        response = openai.ChatCompletion.create(
            model=self._model, messages=valid_messages
        )
        print(response)
        return response.choices[0].message["content"]


class ChatUI:
    """
    A class containing static methods for UI interactions.
    """
    def __init__(self, responder: OpenAIResponder):
        self.responder = responder

    @staticmethod
    def display_messages(column, messages: List[Dict[str, str]]):
        #"""Displays messages in a Streamlit chat."""
        if st.session_state.reversed:
            sorted_messages = reversed(messages)
        else:
            sorted_messages = messages
        for message in sorted_messages:
            with column:
                role = message["role"]
                with st.container():
                    if role == "user":
                        with st.chat_message("user"):
                            st.write(message['content'])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message['content'])

    def get_user_input(self, column, key: str, callback):
        return column.text_input(
            "",
            placeholder="WHo was phone?",
            key=key,
            on_change=callback,
            args=(key,),
        )

    def handle_input_change(self, column_key: str):
        user_input = st.session_state[column_key]
        if user_input:  # Proceed only if the user has entered text
            # Append the user message to the conversation
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Clear the input box after the message is sent
            st.session_state[column_key] = ""

            # Fetch and store the response
            response = self.responder.get_response(st.session_state.messages)
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

    def get_random_score(self):
        # Define the list of possible scores
        possible_scores = list(range(11)) + ['X', 'E']
        
        # Randomly select and return one score
        return random.choice(possible_scores)

    def calculate_average(self, values):
        int_values = []
        count_x = 0
        count_e = 0
        count_int_scores = 0

        for value in values:
            if value == 'X':
                count_x += 1
            elif value == 'E':
                count_e += 1
            else:
                try:
                    # Convert to float and round to integer, count as an integer score
                    int_value = round(float(value))
                    int_values.append(int_value)
                    count_int_scores += 1
                except ValueError:
                    # Handle the case where the value is not a number
                    print(f"Value '{value}' is not a valid number and will be ignored.")

        # Calculate the average if the list of integers is not empty
        average = sum(int_values) / len(int_values) if int_values else None

        return average, count_x, count_e, count_int_scores

    def _evaluate_principle(self, response: str, principle: str) -> int:
        """Evaluates a single safety principle."""

        prompt = safeguard_assessment(response, principle)

        
        #print("Prompt: ", prompt)
        #model_response = self._responder.get_response(
        #    [{"role": "system", "content": prompt}]
        #)
        
        #model_response = "score: {11}"
        #print("SAI assessment: ", model_response)
        #score = extract_score_from_evaluation(model_response)
        score = self.get_random_score()
        return score # as a char

    def get_safety_scores(self, response: str) -> Dict[str, int]:
        """Evaluates safety principles for a given response."""
        return {
            principle['description']: self._evaluate_principle(response, principle['description'])
            for principle in self._safety_principles
        }

    def save_safety_scores(self, scores: Dict[str, int]):
        """Saves the safety scores to a JSON file."""
        with open(self._scores_file_path, "w") as file:
            json.dump(scores, file, indent=4)

    def display_safeguard_evaluation(self, column, response: str):
        """Displays and saves the safety evaluation for a response."""
        safety_scores = self.get_safety_scores(response)

        print("safety scores: ", safety_scores)

        # Calculate the overall score
        overall_score, count_x, count_e, count_int_scores = self.calculate_average(list(safety_scores.values()))

        with column:
            # Construct message with individual scores
            scores_message = ""

            items = list(safety_scores.items())  # Convert to list to get indices
            num_items = len(items)  # Total number of items

            for index, (principle, score) in enumerate(items):

                # We set the border radius depending on the position
                if index == 0:
                    border_radius = (8, 8, 0, 0)
                elif index == num_items - 1:
                    border_radius = (0, 0, 8, 8)
                else:
                    border_radius = 0

                # We set the color of the principle depending on the score
                if score == 'X':
                    color = f"rgb(120, 120, 120)"
                    score_display = "not applicable"
                elif score == 'E':
                    color = f"rgb(0, 0, 0)"
                    score_display = "value error"
                else:
                    color = get_color(score)
                    score_display = score

                scores_message += text_to_html(f"{principle}<br> Score: {score_display}<br>", 
                                                    background_color = color,
                                                    margin = 0,
                                                    border_radius = border_radius)


            # Construct message with overall score
            if overall_score is not None:
                overall_color = get_color(overall_score)
                overall_score_display = f"{overall_score:.2f}"
            else:
                overall_color = f"rgb(0, 0, 0)"
                overall_score_display = "Not Available"

            overall_message = text_to_html(f"Overall Score: {overall_score_display}", 
                                                background_color = overall_color,
                                                strong = True,
                                                margin = (0, 0, 20, 0),
                                                border_radius = 8)


            st.markdown(overall_message, unsafe_allow_html=True)
            st.markdown(scores_message, unsafe_allow_html=True)
            


        # Save the safety scores
        self.save_safety_scores(safety_scores)

def toggle_msg_order():
    st.session_state.reversed = not st.session_state.reversed

def main():
    st.set_page_config(page_title="examine|AI", 
                       page_icon="🧞‍♂️", 
                       layout="wide", 
                       menu_items={
                           'Get Help': 'https://www.betterhelp.com/',
                           'Report a bug': "https://wikiroulette.co/",
                           'About': "### ExamineAI 2023"
    }
                       )
    st.session_state.theme = "dark"

    col1, col2 = st.columns([1, 8])  # Adjust the ratio as needed
    with col1:
        st.image("logo.png", width=130)
        
    with col2:
        st.title("examine|AI")
        st.subheader("Your assistant for trustworthy conversations with AI")

    st.write('<div style="margin-top: 5em;"></div>', unsafe_allow_html=True)

    # Define SafeguardAI object
    responder = OpenAIResponder(api_key=openai.api_key)
    safeguard_ai = SafeguardAI(responder)
    
    # Initialize session state
    MessageStorage()  # Ensures messages are initialized in session state
    if "last_ai_response" not in st.session_state:
        st.session_state.last_ai_response = None
    if "evaluate_safeguard" not in st.session_state:
        st.session_state.evaluate_safeguard = False

    if 'reversed' not in st.session_state:
        st.session_state.reversed = True

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Primary AI")
        subcol1, subcol2 = col1.columns([1, 3])
        model_id = subcol1.selectbox("Selected Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                                                        "gpt-4", "gpt-4-32k"])
        
        subcol2.write('<div style="margin-top: 1.84em;"></div>', unsafe_allow_html=True)
        subcol2.button("Toggle order", on_click=toggle_msg_order)
        # Define ChatUI object
        primary_AI_responder = OpenAIResponder(api_key=openai.api_key, model = model_id)
        chat_ui = ChatUI(responder=primary_AI_responder)
        
        chat_ui.get_user_input(col1, "user_input_primary_ai", chat_ui.handle_input_change)
        chat_ui.display_messages(col1, st.session_state.messages)
        

    # Safeguard AI column
    with col2:
        st.subheader("Safeguard AI")
        if st.session_state.last_ai_response:
            if col2.button("Evaluate", key="evaluate"):
                st.session_state.evaluate_safeguard = True
                st.experimental_rerun()

        if st.session_state.evaluate_safeguard:
            safeguard_ai.display_safeguard_evaluation(
                col2, st.session_state.last_ai_response
            )



if __name__ == "__main__":
    main()
