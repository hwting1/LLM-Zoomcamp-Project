import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from utils import HybridSearchEngine

model_name = "multi-qa-distilbert-cos-v1"
index_name = "medical-questions"
search_engine = HybridSearchEngine(model_name, index_name)

df_ground_truth = pd.read_csv('ground-truth-data.csv')
ground_truth = df_ground_truth.to_dict(orient='records')

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return round(cnt / len(relevance_total), 4)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        found = False
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                found = True
                break
        if not found:
            total_score += 0

    return round(total_score / len(relevance_total), 4)

def evaluate(ground_truth, search_engine, search_method, size=5):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        query = q['question']
        results = search_engine.search(query, size, search_method)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

keyword_results = evaluate(ground_truth, search_engine, "keyword")
knn_results = evaluate(ground_truth, search_engine, "knn")
hybrid_results = evaluate(ground_truth, search_engine, "hybrid")
hybrid_rrf_results = evaluate(ground_truth, search_engine, "hybrid with rrf")

headers = ["search method", "hit rate", "mean reciprocal rank"]

data = [
    ["keyword search", keyword_results["hit_rate"], keyword_results["mrr"]],
    ["knn search", knn_results["hit_rate"], knn_results["mrr"]],
    ["hybrid search", hybrid_results["hit_rate"], hybrid_results["mrr"]],
    ["hybrid search with reciprocal rank fusion", hybrid_rrf_results["hit_rate"], hybrid_rrf_results["mrr"]]
]

print(tabulate(data, headers, tablefmt="grid", colalign=("left", "center", "center")))