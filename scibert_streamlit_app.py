import streamlit as st
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load SciBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_text(text, max_words=400):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def analyze_text(text, keywords):
    blocks = split_text(text)
    results = {kw: [] for kw in keywords}

    for block in blocks:
        block_vec = get_embedding(block)
        for kw in keywords:
            kw_vec = get_embedding(kw)
            sim = cosine_similarity(block_vec, kw_vec)[0][0]
            results[kw].append(sim)

    avg_results = {kw: sum(results[kw]) / len(results[kw]) for kw in keywords}
    return avg_results

def save_results_txt(results_dict):
    output = ""
    for filename, results in results_dict.items():
        output += f"Results for {filename}:\n"
        for kw, score in results.items():
            output += f"  {kw}: {score:.2f}\n"
        output += "\n"
    return output

# Streamlit UI
st.title("SciBERT PDF Analyzer (Web Version)")
st.write("Upload one or more scientific PDFs and enter keywords to check topic similarity.")

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
keywords_input = st.text_input("Enter keywords (comma-separated):", "machine learning, cancer, climate change, disease")

if st.button("Analyze"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    elif not keywords_input.strip():
        st.warning("Please enter at least one keyword.")
    else:
        keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
        results_all = {}

        for file in uploaded_files:
            with st.spinner(f"Analyzing {file.name}..."):
                text = extract_text_from_pdf(file)
                results = analyze_text(text, keywords)
                results_all[file.name] = results

        st.success("Analysis complete!")

        # Show results
        for file_name, results in results_all.items():
            st.subheader(f"Results for {file_name}")
            st.write({k: round(float(v), 2) for k, v in results.items()})

            # Show chart
            fig, ax = plt.subplots()
            ax.bar(results.keys(), results.values())
            ax.set_ylim(0, 1)
            ax.set_ylabel("Similarity")
            ax.set_title(file_name)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Save to text
        txt_results = save_results_txt(results_all)
        st.download_button(
            label="Download results as .txt",
            data=txt_results,
            file_name="scibert_results.txt",
            mime="text/plain"
        )
