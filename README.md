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

curl -X 'GET' \
  'http://0.0.0.0:8000/health' \
  -H 'accept: application/json';


curl -X 'GET' \
  'http://localhost:8000/agent/models' \
  -H 'accept: application/json';


curl -X 'POST' \
  'http://0.0.0.0:8000/agent/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "qwen3:0.6b",
  "prompt": "AI Agents",
  "amount": 1
}';

curl -X 'POST'   'http://0.0.0.0:8000/agent/create_book'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "model":  "qwen3:0.6b",
  "prompt": "AI Agents",
  "amount": 1
}'

```

## api endpoints

### check openapi http://0.0.0.0:8000/docs

![api endpoints](https://raw.githubusercontent.com/santenova/any-api/refs/heads/main/api.png)


## serving

  https://code-agent-1c8a6bd3.data.app/Home

  testing credid card number to get true paywalls
  4111 1111 1111 1111
