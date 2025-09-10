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



## Make sure you run `ollama serve` (this is OpenAI-compatible http://ollama:11434)


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
