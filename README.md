# Final Project for 'LLM-Zoomcamp'

## Introduction

This project utilizes Retrieval-Augmented Generation (RAG) to create a medical assistant capable of answering questions by retrieving relevant information from a medical dataset on Hugging Face. By combining the power of retrieval mechanisms with generative models, the assistant provides contextually accurate and informative responses.

**Important Note**: This medical assistant is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for any medical concerns.

## Project Details

- Dataset: [keivalya/MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
- LLM: [Gemma2 2B](https://ollama.com/library/gemma2)
- Embedding model: [multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1)
- Database: Elasticsearch

## Prerequisites

- Python 3.10 or higher
- Docker

## How to execute it?

1. Create a Python virtual environment and run `pip install -r requirements.txt` to install the required dependencies.

2. Optionally, you can download preprocessed data from my [Google Drive](https://drive.google.com/drive/folders/1BUn4VOIr4dEYbqaAO7gq19fnytYOwIYT?usp=sharing)

3. Run `./start.sh` to start the Docker containers in detached mode, wait for them to initialize, pull the necessary model into the `ollama` container, and then execute the `index_document.py` script to index your documents.

4. Run `streamlit run app.py`, and you can start asking questions to the medical assistant!

5. Optionally, you can run `python retrieval_evaluation.py` to evaluate the retrieval results.

## **Notes**

- I have implemented four types of search methods: keyword, KNN, hybrid, and hybrid with reciprocal rank fusion (document reranking). You can choose your preferred method when using the app, and you can also specify the number of search results to be returned.
- I used only a small portion of the data to generate the ground truth for retrieval evaluation.

**Warning**: This project has only been tested on Ubuntu 22.04. Compatibility with other operating systems is not guaranteed.

**Below is a preview of the application interface:**

![App Interface](app.png)

## Retrieval evaluation results

| Search Method                             | Hit Rate   | Mean Reciprocal Rank |
|-------------------------------------------|------------|----------------------|
| Keyword Search                            | 0.5842     | 0.4132               |
| KNN Search                                | **0.8406** | **0.6795**           |
| Hybrid Search                             | 0.6190     | 0.4372               |
| Hybrid Search with Reciprocal Rank Fusion | 0.8371     | 0.6218               |

