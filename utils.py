from tqdm import tqdm
import hashlib
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class HybridSearchEngine:

    def __init__(self, model_name, index_name, k=60):

        self._index_name = index_name
        self.es_client = Elasticsearch('http://localhost:9200')
        assert self.es_client.indices.exists(index=self._index_name), f"Index '{self._index_name}' does not exist in Elasticsearch."
        self._embedding_model_name = model_name
        self.embedding_model = SentenceTransformer(self._embedding_model_name, device="cuda")

        self._keyword_query = {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": None,
                        "fields": ["Question", "Answer"],
                        "type": "best_fields",
                        "boost": 0.5
                    }
                },
            }
        }
        self._knn_query = {
            "field": "question_answer_vector",
            "query_vector": None,
            "k": 20,
            "num_candidates": 1000,
            "boost": 0.5
        }
        self.k = k

    @property
    def embedding_model_name(self):
        return self._embedding_model_name

    @property
    def index_name(self):
        return self._index_name

    def index_document(self, documents):
        for doc in tqdm(documents):
            self.es_client.index(index=self._index_name, document=doc)

    def search(self, query, size, method):

        if method == "keyword":
            results = self.keyword_search(query, size)
        elif method == "knn":
            results = self.knn_search(query, size)
        elif method == "hybrid":
            results = self.hybrid_search(query, size)
        elif method == "hybrid with rrf":
            return self.hybrid_search_with_rrf(query, size)

        result_docs = []
        for hit in results:
            result_docs.append(hit['_source'])

        return result_docs

    def keyword_search(self, query, size=5):

        self._keyword_query["bool"]["must"]["multi_match"]["query"] = query
        keyword_results = self.es_client.search(
            index=self._index_name,
            body={
                "query": self._keyword_query,
                "size": size
            }
        )['hits']['hits']

        return keyword_results

    def knn_search(self, query, size=5):

        query_vector = self.embedding_model.encode(query)
        self._knn_query["query_vector"] = query_vector

        knn_results = self.es_client.search(
            index=self._index_name,
            body={
                "knn": self._knn_query,
                "size": size
            }
        )['hits']['hits']

        return knn_results

    def hybrid_search(self, query, size=5):

        query_vector = self.embedding_model.encode(query)
        self._keyword_query["bool"]["must"]["multi_match"]["query"] = query
        self._knn_query["query_vector"] = query_vector

        hybrid_results = self.es_client.search(
            index=self._index_name,
            body={
                "knn": self._knn_query,
                "query": self._keyword_query,
                "size": size,
            }
        )['hits']['hits']

        return hybrid_results

    def hybrid_search_with_rrf(self, query, size=5):

        query_vector = self.embedding_model.encode(query)
        self._keyword_query["bool"]["must"]["multi_match"]["query"] = query
        self._knn_query["query_vector"] = query_vector

        knn_results = self.es_client.search(
            index=self._index_name,
            body={
                "knn": self._knn_query,
                "size": size * 2
            }
        )['hits']['hits']

        keyword_results = self.es_client.search(
            index=self._index_name,
            body={
                "query": self._keyword_query,
                "size": size * 2
            }
        )['hits']['hits']

        rrf_scores = {}
        # Calculate RRF using vector search results
        for rank, hit in enumerate(knn_results):
            doc_id = hit['_id']
            rrf_scores[doc_id] = self._compute_rrf(rank + 1)

        # Adding keyword search result scores
        for rank, hit in enumerate(keyword_results):
            doc_id = hit['_id']
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += self._compute_rrf(rank + 1)
            else:
                rrf_scores[doc_id] = self._compute_rrf(rank + 1)

        # Sort RRF scores in descending order
        reranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-K documents by the score
        final_results = []
        for doc_id, score in reranked_docs[:size]:
            doc = self.es_client.get(index=self._index_name, id=doc_id)
            final_results.append(doc['_source'])

        return final_results

    def _compute_rrf(self, rank):
        """ Our own implementation of the relevance score """
        return 1 / (self.k + rank)


def build_prompt(query, search_results):
    context_template = """
Q: {question}
A: {text}
""".strip()

    prompt_template = """
You are a medical assistant. Answer the QUESTION based on the CONTEXT from the medical FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

    context = ""
    for doc in search_results:
        context = context + context_template.format(question=doc['Question'], text=doc['Answer']) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def generate_document_id(doc):
    combined = f"{doc['Question Type']}-{doc['Question']}-{doc['Answer'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:10]
    return document_id

