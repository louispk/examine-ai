# README

## DEPRECATED
ALL FURTHER DEVELOPMENT CAN BE FOUND HERE: https://github.com/OpenBioLink/examine-ai


## Overview

This repository contains the code for `examine|AI`, a Streamlit-based web application that provides an interactive chat interface with an AI (built using OpenAI's GPT model). The application includes a safeguard system that evaluates AI responses against safety principles, aiming to foster trustworthy conversations with AI.

## Features

- Interactive chat with an AI model.
- Storage of chat history with timestamps.
- Safety evaluation of AI responses using predefined principles.
- Docker support for containerization.

## Requirements

- Python 3.x
- Docker (for containerization)
- An OpenAI API key (required for accessing the AI model)

## Installation

Before running the application, ensure you have Docker installed and running on your machine.

### Building the Docker Container

1. Clone the repository to your local machine.
2. Navigate to the cloned directory.
3. Build the Docker image:

```sh
docker build -t examine_ai .
```

### Running the Docker Container

After the image is built, you can run the application inside a Docker container:

```sh
docker run -p 8501:8501 examine_ai
```

The application will be available at `http://localhost:8501`.

## Usage

### Set Up Environment Variable

Before you start using the application, you must set up your OpenAI API key as an environment variable. If running locally without Docker, you can set the environment variable as follows:

```sh
export OPENAI_API_KEY='your_api_key_here'
```

If running with Docker, ensure to pass the API key as an environment variable to the Docker container:

```sh
docker run -e OPENAI_API_KEY='your_api_key_here' -p 8501:8501 examine_ai
```

### Interacting with the Application

- Access the web interface via your browser.
- Type your message into the text input field to start the conversation.
- Use the "Evaluate" button to trigger the safety evaluation of the last AI response.

### AI Safeguard Evaluation

The application will provide an overall score based on safety principles. The scores for individual principles are also displayed for transparency.

## File Descriptions

- `Dockerfile`: Contains all the commands required to build the Docker image.
- `requirements.txt`: Lists all the Python libraries that the app depends on.
- `chat_history.json`: This file is automatically generated and will contain the chat history with timestamps.
- `core_principles.json`: A JSON file that should contain the safety principles for the AI responses to be evaluated against.
- `safety_scores.json`: This file is automatically generated and will store the safety scores of AI responses.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change or add.

## License

Not yet determined.

---

Please note that this application requires a valid OpenAI API key to function. Usage of the API, including any costs incurred, is subject to OpenAI's usage policies and pricing.
