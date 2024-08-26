from datasets import load_dataset
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from utils import generate_document_id

index_name = "medical-questions"
model_name = "multi-qa-distilbert-cos-v1"
model = SentenceTransformer(model_name, device="cuda")

try:
    with open("Medical-QA.json") as f:
        documents = json.load(f)

except FileNotFoundError:
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]
    df = data.to_pandas()
    df = df.rename(columns={"qtype": "Question Type"})
    documents = df.to_dict(orient="records")
    for doc in tqdm(documents):
        qa_text = f"{doc['Question']} {doc['Answer']}"
        embed = model.encode(qa_text)
        doc["question_answer_vector"] = embed.tolist()
        doc["id"] = generate_document_id(doc)

    with open("Medical-QA.json", "w") as f:
        json.dump(documents, f, indent=4)

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "Question": {"type": "text"},
            "Answer": {"type": "text"},
            "Question Type":{"type": "keyword"},
            "id": {"type": "keyword"},
            "question_answer_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

es_client = Elasticsearch('http://localhost:9200')
if not es_client.indices.exists(index=index_name):
    es_client.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)