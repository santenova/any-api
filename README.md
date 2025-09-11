# on localhost

## Make sure you run `ollama serve` (this is OpenAI-compatible http://localhost:11434)


```
cat .env
OLLAMA_BASE=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
```



### Install with uv

```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python ./main.py
```

### Install with pip

```sh
python3.10 -m venv venv
source venv/bin/activate
python3.10 -m pip install -r requirements.txt
python3.10 main.py
```



## Make sure you env is ready for docker


```
cat .env
OLLAMA_BASE=http://ollama:11434
OLLAMA_MODEL=qwen3:0.6b
```


### with docker

```
docker build -t any-api .
docker run -p 8000:8000 -t any-api

```

## Testing

```
# api health
curl -X 'GET' \
  'http://0.0.0.0:8000/health' \
  -H 'accept: application/json';


# available models via omalla
curl -X 'GET' \
  'http://localhost:8000/agent/models' \
  -H 'accept: application/json';


# basic chat
curl -X 'POST' \
  'http://0.0.0.0:8000/agent/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen3:0.6b",
  "prompt": "AI Agents",
  "amount": 1
}';

# basic chat
curl -X 'POST'   'http://0.0.0.0:8000/agent/create_book'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "model":  "qwen3:0.6b",
  "prompt": "AI Agents",
  "amount": 1
}';

# shows db based chat rooms
curl -X 'GET' \
  'http://0.0.0.0:8000/rooms/sessions' \
  -H 'accept: application/json';

# adds db based chat room
curl -X 'POST' \
  'http://0.0.0.0:8000/rooms/new_session' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Product Creation",
  "model_name": "Gemma2:latest"
}';

# start chating in room
curl -X 'POST' \
  'http://0.0.0.0:8000/rooms/sessions/1/messages' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "create all required product information from this text:Model Meter The ultimate tool for evaluating model accuracy and performance",
  "stream": false
}';

# download room as json file
curl -X 'GET' \
  'http://0.0.0.0:8000/rooms/sessions/1/export' \
  -H 'accept: application/json'

```

## api endpoints

### check openapi http://0.0.0.0:8000/docs

![api endpoints](https://raw.githubusercontent.com/santenova/any-api/refs/heads/main/api.png)


## serving

  https://code-agent-1c8a6bd3.base44.app/

  testing credid card number to get true paywalls
  4111 1111 1111 1111
