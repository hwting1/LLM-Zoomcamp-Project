from openai import OpenAI
import streamlit as st
from utils import build_prompt, HybridSearchEngine

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

model_name = "multi-qa-distilbert-cos-v1"
index_name = "medical-questions"
search_engine = HybridSearchEngine(model_name, index_name)

def rag(query, size, method):
    search_results = search_engine.search(query, size, method)
    prompt = build_prompt(query, search_results)
    response = client.chat.completions.create(
        model='gemma2:2b',
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer

# Streamlit UI
st.title("Medical FAQ Assistant")

# User input for the question
query = st.text_input("Enter your question:")

# Input field for the number of results
size = st.number_input("Enter the number of results to retrieve:", min_value=1, step=1, max_value=20, value=5)

# Dropdown to select search type
search_method = st.selectbox(
    "Select the search method:",
    ('keyword', 'knn', 'hybrid', 'hybrid with rrf'),
    index=3
)

# Button to trigger RAG process
if st.button("Enter"):
    if query and isinstance(size, int) and size > 0:
        answer = rag(query, size, search_method)
        st.markdown("### Answer:")
        st.write(answer)
    else:
        st.write("Please enter a valid question and a positive integer for the number of results.")