#!/bin/bash
docker compose up -d

sleep 20

docker exec -it ollama ollama pull gemma2:2b

python index_document.py