import streamlit as st
import openai
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import random
from utils import parse_evaluation, get_color, text_to_html
from prompts import safeguard_assessment
import time

# Make sure to set the OPENAI_API_KEY in your environment variables.
openai.api_key = os.getenv('OPENAI_API_KEY')

placeholder_eval = False
placeholder_response = True


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
        
        return response.choices[0].message["content"]


class ChatUI:
    """
    A class containing static methods for UI interactions.
    """
    def __init__(self, responder: OpenAIResponder):
        self.responder = responder

    @staticmethod
    def display_messages(column, messages: List[Dict[str, str]]):
        print("displaying messages")
        print(messages)
        #"""Displays messages in a Streamlit chat."""
        if st.session_state.reversed:
            sorted_messages = reversed(messages)
        else:
            sorted_messages = messages
        with column:
            for message in sorted_messages:
                role = message["role"]
                if role == "user":
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(message['content'])

    def get_user_input(self, column, key: str, callback):
        return column.text_input(
            "Input",
            label_visibility = "collapsed",
            placeholder="Message chatbot...",
            key=key,
            on_change=callback,
            args=(key,),
        )

    def process_user_input(self, column_key: str):
        user_input = st.session_state[column_key]
        if user_input:  # Proceed only if the user has entered text
            # Append the user message to the conversation
            print("user input saved")
            st.session_state.messages.append({"role": "user", "content": user_input})
            # Clear the input box after the message is sent
            st.session_state[column_key] = ""
            # Store the messages using MessageStorage
            message_storage = MessageStorage()
            message_storage.store_message({"role": "user", "content": user_input})

            # we remove the last evaluation on new user input
            st.session_state.evaluate_safeguard = False
            st.session_state.eval_done = False
            
            # we hide the user input until we receive the response
            st.session_state.show_user_input = False

            # we need to process the user input we received
            st.session_state.user_input_to_be_processed = True

    def process_AI_response(self, column_key: str):
        print("processing AI response")

        # Fetch and store the response
        if not placeholder_response:
            response = self.responder.get_response(st.session_state.messages)
        else:
            time.sleep(2)
            response = "Test Response Please Ignore 420"
        print("AI response saved")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state["last_ai_response"] = response

        # Store the messages using MessageStorage
        message_storage = MessageStorage()
        message_storage.store_message({"role": "assistant", "content": response})

        # We reset this
        st.session_state.user_input_to_be_processed = False
        # we received response
        st.session_state.AI_response_received = True
        # we deactivate the spinner 
        st.session_state.waiting_for_AI_response = False
        # we show the user input again
        st.session_state.show_user_input = True

        print("waiting_for_AI_response = False")
        print("show_user_input = True")


        print("response processed")


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

    def _evaluate_principle(self, response: str, principle: str) -> (str, str):
        """Evaluates a single safety principle."""

        if placeholder_eval:
            return ("assessment", self.get_random_score())
        else:
            prompt = safeguard_assessment(response, principle)

            model_response = self._responder.get_response(
                [{"role": "system", "content": prompt}]
            )
        
            score, assessment = parse_evaluation(model_response)
            return (score, assessment) 


    def get_safety_scores(self, response: str) -> Dict[str, Tuple[str, str]]:
        """Evaluates safety principles for a given response."""
        return {
            principle['description']: (self._evaluate_principle(response, principle['description']))
            for principle in self._safety_principles
        }

    def save_safety_scores(self, scores: Dict[str, int]):
        """Saves the safety scores to a JSON file."""
        with open(self._scores_file_path, "w") as file:
            json.dump(scores, file, indent=4)
        st.session_state.evals = scores

    def display_safeguard_evaluation(self, column, response: str):
        """Displays and saves the safety evaluation for a response."""
        
        # If evaluate_safeguard is False we use stored values
        # Otherwise if eval_done is True we also used stored values
        if st.session_state.evaluate_safeguard:
            if not st.session_state.eval_done:
                safety_scores = self.get_safety_scores(response)
            else:
                safety_scores = st.session_state.evals
        else:
            safety_scores = st.session_state.evals

        # Calculate the overall score
        overall_score, count_x, count_e, count_int_scores = self.calculate_average(s for (a, s) in list(safety_scores.values()))

        # Construct message with overall score
        if overall_score is not None:
            overall_color = get_color(overall_score)
            if overall_score >= 8:
                _overall_color = "green"
            elif overall_score < 8 and overall_score >= 5:
                _overall_color = "orange"
            else:
                _overall_color = "red"
            overall_score_display = f"{overall_score:.2f}"
        else:
            overall_color = f"rgb(75,0,130)"
            _overall_color = "violet"
            overall_score_display = "Not Available"

        overall_message = text_to_html(f"Overall Score: {overall_score_display}", 
                                            background_color = overall_color,
                                            strong = True,
                                            margin = (0, 0, 20, 0),
                                            border_radius = 8)

        if st.session_state.expandable:
            st.markdown(":"+ _overall_color + "[" + f"Overall Score: {overall_score_display}" + "]")

        with column:
            # Construct message with individual scores
            scores_message = ""

            items = list(safety_scores.items())  # Convert to list to get indices
            num_items = len(items)  # Total number of items

            for index, (principle, (a, score)) in enumerate(items):

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
                    _color = "grey"
                    score_display = "not applicable"
                elif score == 'E':
                    color = f"rgb(75,0,130)"
                    _color = "violet"
                    score_display = "value error"
                else:
                    score = int(score)
                    color = get_color(score)
                    if score >= 8:
                        _color = "green"
                    elif score < 8 and score >= 5:
                        _color = "orange"
                    else:
                        _color = "red"

                    score_display = str(score) + "/10"

                expander_label = ":"+ _color + "[" + f"{principle[:-1]}: " + f"{score_display}" + "]"

                if st.session_state.expandable:
                # Use an expander for each principle
                    with st.expander(
                        expander_label,
                        expanded = False
                        ):
                        st.write(a)

                else:
                    scores_message += text_to_html(f"{principle}<br> Score: {score_display}<br>", 
                                                    background_color = color,
                                                    margin = 0,
                                                    border_radius = border_radius)

            if not st.session_state.expandable:
                st.markdown(overall_message, unsafe_allow_html=True)
                st.markdown(scores_message, unsafe_allow_html=True)


        # Save the safety scores
        self.save_safety_scores(safety_scores)
        st.session_state.eval_done = True
        st.session_state.evaluate_safeguard = False

def toggle_msg_order():
    st.session_state.reversed = not st.session_state.reversed

def expandable_evals():
    st.session_state.expandable = not st.session_state.expandable
    st.session_state.evaluate_safeguard = True

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

    col1, col2 = st.columns([1, 5])  # Adjust the ratio as needed
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
    if 'last_ai_response' not in st.session_state:
        st.session_state.last_ai_response = None
    if 'evaluate_safeguard' not in st.session_state:
        st.session_state.evaluate_safeguard = False
    if 'reversed' not in st.session_state:
        st.session_state.reversed = False
    if 'expandable' not in st.session_state:
        st.session_state.expandable = False
    if 'eval_done' not in st.session_state:
        st.session_state.eval_done = False
    if 'evals' not in st.session_state:
        st.session_state.evals = " "
    if 'user_input_to_be_processed' not in st.session_state:
        st.session_state.user_input_to_be_processed = False
    if 'show_user_input' not in st.session_state:
        st.session_state.show_user_input = True
    if 'disable_user_input' not in st.session_state:
        st.session_state.disable_user_input = False
    if 'waiting_for_AI_response' not in st.session_state:
        st.session_state.waiting_for_AI_response = False
    if 'AI_response_received' not in st.session_state:
        st.session_state.AI_response_received = False

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Primary AI")
        subcol1, subcol2 = col1.columns([1, 2])
        model_id = subcol1.selectbox("Selected Model",
                                     ["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                                      "gpt-4", "gpt-4-32k"],
                                      label_visibility = "collapsed")
        
        #subcol2.button("Toggle order", on_click=toggle_msg_order)
        subcol2.checkbox("Reversed", on_change=toggle_msg_order)
        # Define ChatUI object
        primary_AI_responder = OpenAIResponder(api_key=openai.api_key, model = model_id)
        chat_ui = ChatUI(responder=primary_AI_responder)

        # We put the input below or above the messages depending on the reversed status
        if st.session_state.reversed:
            if st.session_state.waiting_for_AI_response:
                with st.spinner("waiting"):
                    # we draw the convo again here so it is
                    # visible while spinner spins
                    chat_ui.display_messages(col1, st.session_state.messages)
                    # now that we have the user input we process the AI response
                    chat_ui.process_AI_response(col1)
            if st.session_state.show_user_input:
                chat_ui.get_user_input(col1, "user_input_primary_ai_top", chat_ui.process_user_input)
        chat_ui.display_messages(col1, st.session_state.messages)
        if not st.session_state.reversed:
            if st.session_state.waiting_for_AI_response:
                with st.spinner("waiting"):
                    # now that we have the user input we process the AI response
                    chat_ui.process_AI_response(col1)
            if st.session_state.show_user_input:
                chat_ui.get_user_input(col1, "user_input_primary_ai_bottom", chat_ui.process_user_input)

        if st.session_state.user_input_to_be_processed:
            # process the AI response
            st.session_state.waiting_for_AI_response = True
            print("rerun to activate spinner")
            st.rerun()        

        # we do a rerun after receiving the AI response
        # to redraw chat without the chat in the with st.spinner("waiting")
        if st.session_state.AI_response_received:
            st.session_state.AI_response_received = False
            st.rerun();


    # Safeguard AI column
    with col2:
        st.subheader("Safeguard AI")
        subcol1, subcol2 = col2.columns([1, 2])
        if st.session_state.last_ai_response:
            if subcol1.button("Evaluate", key="evaluate"):
                st.session_state.evaluate_safeguard = True
                st.session_state.eval_done = False
            subcol2.checkbox("Expandable", on_change = expandable_evals)

        if st.session_state.evaluate_safeguard:
            safeguard_ai.display_safeguard_evaluation(
                col2, st.session_state.last_ai_response
            )

if __name__ == "__main__":
    main()
