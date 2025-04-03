# kedro-rag-chatbot
This project demonstrates how to use Kedro to create a RAG-based Chatbot.

In this example we will use questions and answers from our [Kedro Slack](https://kedro-org.slack.com/archives/C03RKP2LW64) 
user support channel to create a vector store. Then we create a GenAI-based agent which can query our vector store to 
answer questions about Kedro.

Se the demo on [YouTube](https://www.youtube.com/watch?v=rgmANk-QwYg)

## Setup

1. Clone this project locally
2. Install dependencies

```
pip install -r requirements.txt
```
3. All the necessary data needed for a test run is already placed into `data/01_raw`

## How to run

### 1. Create vector store

```
kedro run -p create_vector_store
```

### 2. Create vector store

```
kedro run -t agent_rag
```
