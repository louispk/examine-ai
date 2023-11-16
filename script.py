import streamlit as st
import openai
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple
import random
from utils import parse_evaluation, get_color, text_to_html, calculate_average, get_random_score
from prompts import safeguard_assessment
import time
from datetime import datetime
import logging

# Make sure to set the OPENAI_API_KEY in your environment variables.
openai.api_key = os.getenv('OPENAI_API_KEY')

# set to True to use placeholder evals
placeholder_eval = False
# set to True to use placeholder response
placeholder_response = False
# set to true to see a toast with response details
show_message_toast = True

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main_log.log', mode='a'),  # Log to this file
        # If you want to log to both file and console, uncomment the next line
        # logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MessageStorage:
    """
    A class for storing chat messages with a timestamp to a file.
    """

    def __init__(self, file_path: str = 'chat_history.json'):
        self._file_path = file_path
        self._initialize_messages()

    def _initialize_messages(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def store_message(self, message: Dict[str, str]):
        """Stores a message dictionary with a timestamp to a JSON file."""
        message_with_timestamp = {'timestamp': datetime.now().isoformat(), **message}
        with open(self._file_path, 'a') as f:
            json.dump(message_with_timestamp, f)
            f.write('\n')


class OpenAIResponder:
    """
    A class to handle responses from OpenAI's GPT model.
    """
    def __init__(self, api_key: str, model: str = 'gpt-3.5-turbo'):
        self._api_key = api_key
        self._model = model

    def get_response(self, messages: List[Dict[str, str]]) -> (str, str):
        """Fetches a response from OpenAI using the given list of message dictionaries."""
        content = ''
        status = 'ERR'
        details = None
        try:
            valid_messages = []
            for msg in messages:
                if msg['role'] in ['system', 'assistant', 'user', 'function']:
                    if status not in msg or msg['status'] == 'OK':
                        valid_messages.append({'role': msg['role'], 'content': msg['content']})

            response = openai.ChatCompletion.create(
                model=self._model, messages=valid_messages
            )

            details = {}
            details['received'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details['finish_reason'] = response.choices[0].finish_reason
            details['created'] = response.created
            details['id'] = response.id 
            details['model'] = response.model
            details['usage'] = response.usage

            status = 'OK' if details['finish_reason'] == 'stop' else 'WARN'
            return response.choices[0].message['content'], status, details

        except openai.error.Timeout as e:
            content = f"OpenAI API request timed out: {e}"
        except openai.error.APIError as e:
            content = f"OpenAI API returned an API Error: {e}"
        except openai.error.APIConnectionError as e:
            content = f"OpenAI API request failed to connect: {e}"
        except openai.error.InvalidRequestError as e:
            content = f"OpenAI API request was invalid: {e}"
        except openai.error.AuthenticationError as e:
            content = f"OpenAI API request was not authorized: {e}"
        except openai.error.PermissionError as e:
            content = f"OpenAI API request was not permitted: {e}"
        except openai.error.RateLimitError as e:
            content = f"OpenAI API request exceeded rate limit: {e}"

        return content, status, details


class ChatUI:
    """
    A class containing static methods for UI interactions.
    """

    @staticmethod
    def get_color(status : str):
        mapping = {
            'OK' : 'white',
            'WARN' : 'orange', 
            'ERR' : 'red'
        }
        color = mapping[status]
        if color is None:
            color = 'violet'
        return color

    @staticmethod
    def print_message(message : str, last_msg : bool):
        color = ChatUI.get_color(message['status'])
        message_placeholder = st.empty()
        full_response = ''
        # we only stream the response once, when we receive it
        if (st.session_state.streaming and 
            last_msg and 
            st.session_state.stream_message and 
            message['status'] != 'ERR'):
            for chunk in message['content'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                #message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response + '▌')
        if color != 'white':
            message_placeholder.markdown(":"+ color + "[" + message['content'] + "]")
        else:
            message_placeholder.markdown(message['content'])

    @staticmethod
    def display_messages(column, messages: List[Dict[str, str]]):
        #"""Displays messages in a Streamlit chat."""
        if st.session_state.reversed:
            sorted_messages = reversed(messages)
        else:
            sorted_messages = messages
        msg_ct = len(sorted_messages)
        with column:
            for idx, message in enumerate(sorted_messages):
                role = message['role']
                if role == 'user':
                    with st.chat_message('user'):
                        st.write(message['content'])
                elif role == 'assistant':
                    with st.chat_message('assistant'):
                        # we only want to stream the last received AI repsonse
                        ChatUI.print_message(message, idx == msg_ct - 1)
                else:
                    with st.chat_message(role):
                        st.write(message['content'])
    
    @staticmethod
    def display_chat_input(col, responder, top):
        if st.session_state.reversed == top:
            if st.session_state.waiting_for_AI_response:
                with st.spinner('AI is thinking'):
                    # if we are above the messages we draw the convo 
                    # again here so it is visible while spinner spins
                    #if top:
                    #    ChatUI.display_messages(col, st.session_state.messages)
                    if st.session_state.show_user_input:
                        key = 'my_form_while_spinning_' + 'top' if top else 'bottom'
                        ChatUI.handle_user_input(key)
                    # now that we have the user input we process the AI response
                    ChatUI.process_AI_response(responder)
            if st.session_state.show_user_input:
                key = 'my_form_' + ('top' if top else 'bottom')
                ChatUI.handle_user_input(key)

    @staticmethod
    def handle_user_input(key : str):
        with st.form(key=key):
            subcol1, subcol2 = st.columns([3, 1])
            user_input = subcol1.empty()
            st.markdown("""
                <style>
                div.stButton > button {
                    margin: 0 auto; /* Center the button */
                    display: block; /* Necessary for centering */
                }
                </style>""", unsafe_allow_html=True)
            # A submit button to handle the form submission
            submit_button = subcol2.form_submit_button(
                label='Submit',
                disabled=st.session_state.disable_user_input
            )

            user_input_key = key + '_' + 'user_input_field'

            # Check if the form has been submitted
            if submit_button:
                ChatUI.process_user_input(user_input_key)
                st.session_state.user_input = ''

            user_input.text_input(
                'Input',
                label_visibility = 'collapsed',
                placeholder='Message chatbot...',
                key=user_input_key,
                disabled=st.session_state.disable_user_input,
            )

    @staticmethod
    def process_user_input(key: str):
        user_input = st.session_state[key]
        if user_input:  # Proceed only if the user has entered text
            # Append the user message to the conversation
            st.session_state.messages.append({
                'role': 'user', 
                'content': user_input
            })
            # Clear the input box after the message is sent
            st.session_state[key] = ''
            # Store the messages using MessageStorage
            message_storage = MessageStorage()
            message_storage.store_message({
                'role': 'user', 
                'content': user_input
            })
            
            # we hide the user input until we receive the response
            st.session_state.show_user_input = False
            # we need to process the user input we received
            st.session_state.user_input_to_be_processed = True
            # we disable checkboxes
            st.session_state.disable_user_input = True
            # we disable the eval button
            st.session_state.disable_eval_button = True

    def process_AI_response(responder):
        response = status = details = None
        # Fetch and store the response
        if not placeholder_response:
            response, status, details = responder.get_response(st.session_state.messages)
        else:
            time.sleep(2)
            response = ''
            for i in range (100) :
                response += 'testing '
            status = random.choice(['OK', 'WARN', 'ERR'])

        # we add the message to the state
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': response, 
            'status' : status
        })
        st.session_state.last_ai_response = response
        st.session_state.last_ai_response_details = details

        # Store the messages using MessageStorage
        message_storage = MessageStorage()
        message_storage.store_message({
            'role': 'assistant', 
            'content': response, 
            'status' : status
            })

        # we reset this
        st.session_state.user_input_to_be_processed = False
        # we received response
        st.session_state.AI_response_received = True
        # we deactivate the spinner 
        st.session_state.waiting_for_AI_response = False

class SafeguardAI:
    """
    A class to evaluate and display AI response safety based on predefined principles.
    """

    def __init__(
        self,
        responder: OpenAIResponder,
        principles_file_path: str = 'core_principles.json',
        scores_file_path: str = 'safety_scores.json',
    ):
        self._responder = responder
        self._principles_file_path = principles_file_path
        self._scores_file_path = scores_file_path
        self._safety_principles = self._load_safety_principles()

    def _load_safety_principles(self) -> List[str]:
        """Loads safety principles from a JSON file."""
        with open(self._principles_file_path, 'r') as file:
            principles = json.load(file)
        return principles.get('principles', [])

    def _evaluate_principle(self, response: str, principle: str) -> (str, str):
        """Evaluates a single safety principle."""

        if placeholder_eval:
            time.sleep(0.5)
            return ("assessment ... ", get_random_score())
        else:
            prompt = safeguard_assessment(response, principle)

            content, status, details = self._responder.get_response(
                [{'role': 'system', 'content': prompt}]
            )
            logger.info(content)
        
            score, assessment = parse_evaluation(content)
            return (score, assessment) 

    def _get_safety_scores(self, response: str) -> Dict[str, Tuple[str, str]]:
        """Evaluates safety principles for a given response."""
        return {
            principle['description']: (self._evaluate_principle(response, principle['description']))
            for principle in self._safety_principles
        }

    def _save_safety_scores(self, scores: Dict[str, Tuple[str, str]]):
        """Saves the safety scores to a JSON file."""
        with open(self._scores_file_path, 'w') as file:
            json.dump(scores, file, indent=4)
        st.session_state.evals = scores

    def _display_expandable(self, column):
        # we retrieve evaluations
        safety_scores = st.session_state.evals

        # calculate the overall score
        overall_score, count_x, count_e, count_int_scores = calculate_average(s for (a, s) in list(safety_scores.values()))

        # Construct message with overall score
        if overall_score is not None:
            if overall_score >= 8:
                overall_color = 'green'
            elif overall_score < 8 and overall_score >= 5:
                overall_color = 'orange'
            else:
                overall_color = 'red'
            overall_score_display = f"{overall_score:.2f}"
        else:
            overall_color = 'violet'
            overall_score_display = "Not Available"

        column.markdown(':'+ overall_color + '[' + f"Overall Score: {overall_score_display}" + ']')

        # we ennumerate over all principles and draw them one by one
        for index, (principle, (a, score)) in enumerate(list(safety_scores.items())):
            # We set the color of the principle depending on the score
            if score == 'X':
                color = 'grey'
                score_display = 'not applicable'
            elif score == 'E':
                color = 'violet'
                score_display = 'value error'
            else:
                score = int(score)
                if score >= 8:
                    color = 'green'
                elif score < 8 and score >= 5:
                    color = 'orange'
                else:
                    color = 'red'

                score_display = str(score) + '/10'

            expander_label = ':'+ color + '[' + f"{principle[:-1]}: " + f"{score_display}" + ']'
            # Use an expander for each principle
            with column.expander(
                expander_label,
                expanded = False
                ):
                st.write(a)
        

    def _display_normal(self, column):

        # we retrieve evaluations
        safety_scores = st.session_state.evals

        # calculate the overall score
        overall_score, count_x, count_e, count_int_scores = calculate_average(s for (a, s) in list(safety_scores.values()))

        # Construct message with overall score
        if overall_score is not None:
            overall_color = get_color(overall_score)
            overall_score_display = f"{overall_score:.2f}"
        else:
            overall_color = f'rgb(75,0,130)'
            overall_score_display = 'Not Available'

        html_scores = text_to_html(
            f"Overall Score: {overall_score_display}", 
            background_color = overall_color,
            strong = True,
            margin = (0, 0, 20, 0),
            border_radius = 8)

        items = list(safety_scores.items())
        num_items = len(items)

        # we ennumerate over all principles and draw them one by one
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
                color = f'rgb(120, 120, 120)'
                score_display = 'not applicable'
            elif score == 'E':
                color = f'rgb(75,0,130)'
                score_display = 'value error'
            else:
                score = int(score)
                color = get_color(score)

                score_display = str(score) + '/10'

            # we append html for each principle to html_scores
            html_scores += text_to_html(f"{principle}<br> Score: {score_display}<br>", 
                                                background_color = color,
                                                margin = 0,
                                                border_radius = border_radius)
        # we draw the html to the container
        column.markdown(html_scores, unsafe_allow_html=True)


    def obtain_safeguard_evaluation(self, response: str):
        """Obtains and saves the safety evaluation for a response."""
        safety_scores = self._get_safety_scores(response)

        # Save the safety scores
        self._save_safety_scores(safety_scores)

    def display_safeguard_evaluation(self, column):
        if st.session_state.expandable:
            self._display_expandable(column)
        else:
            self._display_normal(column)


def main():
    st.set_page_config(
        page_title='examine|AI', 
        page_icon='🧞‍♂️', 
        layout='wide', 
        initial_sidebar_state='expanded', # 'auto', 'expanded', 'collapsed'
        menu_items={
            'Get Help': "https://examine.dev/#building-ai-meta-systems",
            'Report a bug': "https://examine.dev/#team",
            'About': "### ExamineAI 2023"
        }
    )
    st.session_state.theme = 'dark'

    # Ensures messages are initialized in session state
    MessageStorage()  
    # Initialize session state
    default_session_state = {
          'last_ai_response': None,
          'last_ai_response_details' : None,
          'evaluate_pressed': False,
          'disable_eval_button': False,
          'obtain_evaluation': False,
          'display_evaluation': False,
          'reversed': False,
          'expandable': False,
          'evals': None,
          'user_input_to_be_processed': False,
          'show_user_input': True,
          'disable_user_input': False,
          'waiting_for_AI_response': False,
          'AI_response_received': False,
          'show_message_toast' : False,
          'streaming' : False,
          'stream_message' : True,
          'model_id' : 'gpt-3.5-turbo',
          'evaluations_received' : False
          }
    for key, default_value in default_session_state.items():
          if key not in st.session_state:
              st.session_state[key] = default_value

    with st.sidebar:
        st.title('Settings')
        selected_model_id = st.selectbox(
            'Selected Model',
            ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k',
             'gpt-4', 'gpt-4-32k'],
             key='select_model',
             label_visibility='visible',
             disabled=st.session_state.disable_user_input)

        if selected_model_id != st.session_state.model_id:
            st.session_state.model_id = selected_model_id
            st.toast('Primary AI model set to ' + st.session_state.model_id)
        
        if st.checkbox(
            'Reversed', 
            key='reversed_checkbox',
            disabled=st.session_state.disable_user_input,
            value = st.session_state.reversed,
        ):
            st.session_state.reversed = True
        else:
            st.session_state.reversed = False

        if st.checkbox(
            'Streaming (experimental)',
            key='streaming_checkbox',
            disabled=st.session_state.disable_user_input,
            value=st.session_state.streaming,
        ):
            st.session_state.streaming = True
        else:
            st.session_state.streaming = False

        with st.expander('Info', expanded=False):
            if st.session_state.last_ai_response_details is not None:
                for field in st.session_state.last_ai_response_details:
                    st.write(str(field) + ': ' + str(st.session_state.last_ai_response_details[field]))


    col1, col2 = st.columns([1, 5])  # Adjust the ratio as needed
    with col1:
        st.image('logo.png', width=130)
        st.markdown('''
                    <style>
                    button[title="View fullscreen"]{
                        visibility: hidden;}
                    </style>
                    ''',
                    unsafe_allow_html=True)    

    with col2:
        st.title("examine|AI", 
                 anchor=False)
        st.subheader("Your assistant for trustworthy conversations with AI", 
                     anchor=False)

    st.write('<div style="margin-top: 5em;"></div>', unsafe_allow_html=True)

    # Define SafeguardAI object
    responder = OpenAIResponder(api_key=openai.api_key)
    safeguard_ai = SafeguardAI(responder)
    
    col1, col2 = st.columns(2, gap='medium')

    with col1:
        st.subheader('Primary AI', anchor=False)
        
        # Define Primary AI responder
        primary_AI_responder = OpenAIResponder(
            api_key = openai.api_key, 
            model = st.session_state.model_id)

        # We put the input below or above the messages depending on the reversed status
        ChatUI.display_chat_input(col1, primary_AI_responder, top = True)
        ChatUI.display_messages(col1, st.session_state.messages)
        ChatUI.display_chat_input(col1, primary_AI_responder, top = False)    

        
    # Safeguard AI column
    with col2:
        st.subheader('Safeguard AI', anchor=False)
        subcol1, subcol2 = col2.columns([1, 2])
        if st.session_state.last_ai_response is not None:
            if subcol1.button(
                'Evaluate', 
                key='evaluate', 
                disabled=st.session_state.disable_eval_button
            ):
                st.session_state.evaluate_pressed = True
                st.session_state.disable_eval_button = True

            if subcol2.checkbox(
                'Expandable', 
                key='expandable_checkbox',
                disabled=st.session_state.disable_eval_button
            ):
                st.session_state.expandable = True
            else:
                st.session_state.expandable = False

        if st.session_state.obtain_evaluation:
            with st.spinner('Waiting for evaluations'):
                safeguard_ai.obtain_safeguard_evaluation(
                    st.session_state.last_ai_response
                )
                st.session_state.evaluations_received = True

    #
    # Here we handle state

    if st.session_state.evaluations_received:
        st.session_state.evaluations_received = False
        # we reenable user input after evaluations are shown
        st.session_state.disable_user_input = False
        # we reset this abter we obtained the evaluations
        st.session_state.obtain_evaluation = False
        # we can show the evaluation now
        st.session_state.display_evaluation = True
        # we reenable the eval button
        st.session_state.disable_eval_button = False
        st.rerun()


    # we display the evals from the safeguard AI
    if st.session_state.display_evaluation:
        safeguard_ai.display_safeguard_evaluation(col2)

    if st.session_state.user_input_to_be_processed:
        st.session_state.waiting_for_AI_response = True
        st.session_state.display_evaluation = False
        st.rerun()  

    if st.session_state.AI_response_received:
        st.session_state.AI_response_received = False
        #st.session_state.show_message_toast = True
        st.session_state.stream_message = True
        st.rerun()

    if st.session_state.stream_message:
        st.session_state.stream_message = False
        # we reactivate deactivated inputs
        st.session_state.disable_user_input = False
        # we reactivate the eval button
        st.session_state.disable_eval_button = False
        # we show the user text input again
        st.session_state.show_user_input = True
        st.session_state.show_message_toast = True

        st.rerun()

    if st.session_state.show_message_toast:
        st.session_state.show_message_toast = False
        if show_message_toast:
            st.toast('Response received successfully!')

    if st.session_state.evaluate_pressed:
        # we remove eisting evals
        st.session_state.display_evaluation = False
        # we only want one rerun
        st.session_state.evaluate_pressed = False
        # we disbale user input after evaluate is pressed
        st.session_state.disable_user_input = True
        # we do the actual evaluation during next run
        st.session_state.obtain_evaluation = True
        st.rerun()

if __name__ == '__main__':
    main()
