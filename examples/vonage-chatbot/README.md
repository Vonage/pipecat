# Vonage Chatbot (Pipecat)

A real-time voice chatbot built using **Pipecat AI** with **Vonage Audio Connector** over **WebSocket**.
This project streams caller audio to **OpenAI STT**, processes the conversation using an LLM, converts the AI's response to speech via **OpenAI TTS**, and streams it back to the caller in real time. The server exposes a WebSocket endpoint (via **VonageAudioConnectorTransport**) that the Vonage **/connect API** connects to, bridging a live session into the **OpenAI STT → LLM → TTS** pipeline.


## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation and Configuration](#installation-and-configuration)
- [Running the Application](#running-the-application)
- [Testing the Chatbot](#testing-the-chatbot)

## Features

- **Real-time, bidirectional audio** using WebSockets via Vonage Audio Connector
- **OpenAI-powered pipeline:** STT → LLM → TTS
- **Silero VAD** for accurate speech-pause detection
- **Docker support** for simple deployment and isolation

## Requirements

- Python **3.10+**
- A **Vonage(Opentok) account**
- An **OpenAI API Key**
- **ngrok** (or any HTTPS tunnel) for local testing
- Docker (optional)

## Installation and Configuration

1. **Clone the repo and enter it**

    ```sh
    git clone https://github.com/opentok/vonage-pipecat.git
    cd vonage-pipecat/
    ```

2. **Set up a virtual environment** (recommended):

    ```sh
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

3. **Install Pipecat AI (editable mode)**:

    ```sh
    pip install -e ".[openai,websocket,vonage-audio-connector,silero,runner]"
    ```

4. **Install example dependencies**:

    ```sh
    cd examples/vonage-chatbot
    pip install -r requirements.txt
    ```

5. **Create your .env file**:

    ```sh
    cp env.example .env
    ```
    Update .env with your credentials and session ID as mentioned in Steps 6 and 7 below.

6. **Create an Opentok/Vonage Session and Publish a Stream**
    A Session ID is required for the Audio Connector.
    Note: You can use either Opentok or Vonage platform to create the session. Open the Playground (or your own app) to create a session and publish a stream.
    Copy the Session ID and set it in `.env` file:
    ```sh
    VONAGE_SESSION_ID=<paste-your-session-id-here>
    ```
    Always use **credentials from the same project** that created the `sessionId`.

7. **Set the Keys in `.env`**
    If the session was created using the OpenTok (API key + secret), set the following in your `.env`:

    ```sh
    # OpenAI Key (no quotes)
    OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx

    # OpenTok credentials
    VONAGE_API_KEY=YOUR_API_KEY
    VONAGE_API_SECRET=YOUR_API_SECRET

    # Session ID created in Step 6
    VONAGE_SESSION_ID=1_MX4....

    # Leave blank; this is auto-filled after `/connect` API call
    VONAGE_CONNECTION_ID=...

    ```
   If the session was created using the Vonage platform (App ID + Private Key), set the following in your `.env`:

    ```sh
    # Vonage Platform API credentials
    VONAGE_APPLICATION_ID=YOUR_APPLICATION_ID
    VONAGE_PRIVATE_KEY=YOUR_PRIVATE_KEY_PATH

    # Session ID created in Step 6
    VONAGE_SESSION_ID=1_MX4....

    # Leave blank; auto-filled by client.py
    VONAGE_CONNECTION_ID=...

    ```

8. **Install ngrok**:

   Follow the instructions on the [ngrok website](https://ngrok.com/download) to download and install ngrok. You’ll use this to securely expose your local WebSocket server for testing.

9. **Start ngrok to expose the local WebSocket server**:

    **Run in a separate terminal**, start ngrok to tunnel the local server:

    ```sh
    ngrok http 8005
    ```

    You will see something like:

    ```sh
    Forwarding    https://a5db22f57efa.ngrok-free.app -> http://localhost:8005
    ```

    To form the **WSS** URL, replace https:// with wss://.

    Example like for above Forwarding URL below is the wss:// url:

    ```sh
    "websocket": {
        "uri": "wss://a5db22f57efa.ngrok-free.app",
        "audioRate": 16000,
        "bidirectional": true
    }
    ```

## Running the Application

You can run the Chatbot server using the Python or Docker.

### Option 1: Run with Python

    ```sh
    # Make sure your virtualenv is active and you are inside examples/vonage-chatbot
    python server.py
    ```
    The server will start on port 8005 and wait for incoming Audio Connector connections.

### Option 2: Run with Docker

1. **Build the Docker image**:

    ```sh
    docker build -f examples/vonage-chatbot/Dockerfile -t vonage-chatbot .
    ```

2. **Run the Docker container**:
    ```sh
    docker run -it --rm -p 8005:8005 --env-file examples/vonage-chatbot/.env vonage-chatbot
    ```

## Testing the Chatbot

1. Follow the instructions in: `examples/vonage-chatbot/client/README.md`.
2. Run the client program (`connect_and_stream.py`) to invoke the **/connect API**.
3. Once the connection is established, begin speaking in the Vonage Video session. Your audio will be forwarded through the Audio Connector to the Pipecat pipeline processed by OpenAI STT → LLM → TTS and the synthesized response will be sent back into the session. 
4. You will hear the AI’s voice reply in real time, played back as audio from the virtual participant created by the `/connect` API.
