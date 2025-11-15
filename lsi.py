import numpy as np
import pandas as pd
import re
from collections import Counter

# -----------------------
# Common helpers
# -----------------------

def tokenize(text: str):
    """Simple tokenizer: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if t]

def build_term_doc_matrix(texts):
    """
    Build a term-document matrix (terms x docs) using raw counts.
    Returns: A (np.ndarray), vocab (dict term->index), doc_ids (range(len(texts)))
    """
    vocab = {}
    triples = []  # (row, col, value)
    
    for j, doc in enumerate(texts):
        counts = Counter(tokenize(doc))
        for term, c in counts.items():
            if term not in vocab:
                vocab[term] = len(vocab)
            i = vocab[term]
            triples.append((i, j, c))
    
    if not vocab:
        return np.zeros((0, len(texts))), {}, list(range(len(texts)))
    
    A = np.zeros((len(vocab), len(texts)), dtype=float)
    for i, j, c in triples:
        A[i, j] = c
    
    return A, vocab, list(range(len(texts)))

def lsi_svd(A, k):
    """
    Compute rank-k LSI using SVD of term-document matrix A.
    A: terms x docs
    Returns: U_k, S_k, Vt_k
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, len(s))
    U_k = U[:, :k]              # term-concept matrix
    S_k = np.diag(s[:k])        # singular values (k x k)
    Vt_k = Vt[:k, :]            # concept-document matrix
    return U_k, S_k, Vt_k

def project_documents_lsi(S_k, Vt_k):
    """
    Get document vectors in latent space (k-dim).
    Each column of Vt_k corresponds to a document.
    doc_coords = S_k * V_k (k x n_docs)
    """
    return S_k @ Vt_k  # shape: (k, n_docs)

# -----------------------
# LSI from in-memory dataset
# -----------------------

def lsi_from_texts(texts, k=2):
    """
    texts: list of document strings
    """
    A, vocab, doc_ids = build_term_doc_matrix(texts)
    U_k, S_k, Vt_k = lsi_svd(A, k)
    doc_coords = project_documents_lsi(S_k, Vt_k)
    return {
        "A": A,
        "vocab": vocab,
        "doc_ids": doc_ids,
        "U_k": U_k,
        "S_k": S_k,
        "Vt_k": Vt_k,
        "doc_coords": doc_coords   # shape (k, n_docs)
    }

# Example usage (NO CSV):
if __name__ == "__main__" and False:  # set to True to run
    docs = [
        "Information retrieval and web search",
        "Latent semantic indexing using singular value decomposition",
        "Web search and ranking using link analysis",
    ]
    result = lsi_from_texts(docs, k=2)
    print("Term-document matrix shape:", result["A"].shape)
    print("Doc coordinates in 2D LSI space:\n", result["doc_coords"].T)

# -----------------------
# LSI from CSV
# -----------------------

def lsi_from_csv(csv_path, text_col="text", k=2):
    """
    csv_path: path to CSV file
    text_col: name of column containing document text
    """
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    return lsi_from_texts(texts, k=k)

# Example usage (WITH CSV):
if __name__ == "__main__" and False:  # set to True to run
    res = lsi_from_csv("documents.csv", text_col="content", k=3)
    print("Doc coords (k x n_docs):\n", res["doc_coords"])
