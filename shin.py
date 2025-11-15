import numpy as np
import pandas as pd
import random

# -----------------------
# Shingling
# -----------------------

def get_k_shingles(text: str, k: int = 3):
    """
    Return set of k-shingles (as strings) from text.
    Shingles are formed over tokens (word-level shingles).
    """
    tokens = tokenize(text)
    if len(tokens) < k:
        return set()
    shingles = set()
    for i in range(len(tokens) - k + 1):
        sh = " ".join(tokens[i:i+k])
        shingles.add(sh)
    return shingles

def build_shingle_sets(texts, k=3):
    """
    texts: list of document strings
    Returns:
        doc_shingles: list of sets (shingles per doc)
        shingle_to_id: dict mapping shingle -> integer id
    """
    all_shingles = set()
    doc_shingles = []
    for txt in texts:
        s = get_k_shingles(txt, k)
        doc_shingles.append(s)
        all_shingles |= s
    
    shingle_to_id = {sh: idx for idx, sh in enumerate(sorted(all_shingles))}
    # Convert each doc's shingles to ids
    doc_shingles_ids = []
    for sset in doc_shingles:
        doc_shingles_ids.append({shingle_to_id[sh] for sh in sset})
    
    return doc_shingles_ids, shingle_to_id

# -----------------------
# MinHash via random permutations
# -----------------------

def minhash_random_permutations(doc_shingles_ids, num_perm=50, seed=42):
    """
    MinHash signature using explicit random permutations of the shingle universe.
    doc_shingles_ids: list of sets of shingle ids for each document
    Returns: signatures matrix (num_perm x num_docs)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    all_ids = sorted({sid for s in doc_shingles_ids for sid in s})
    n_univ = len(all_ids)
    n_docs = len(doc_shingles_ids)
    
    if n_univ == 0:
        return np.zeros((num_perm, n_docs), dtype=int)
    
    id_index = {sid: i for i, sid in enumerate(all_ids)}
    
    signatures = np.full((num_perm, n_docs), fill_value=np.inf)
    
    for p in range(num_perm):
        perm = list(range(n_univ))
        random.shuffle(perm)
        # position_of[universe_id] = position in perm
        position_of = {all_ids[i]: perm[i] for i in range(n_univ)}
        
        for d, sset in enumerate(doc_shingles_ids):
            if not sset:
                continue
            min_pos = min(position_of[sid] for sid in sset)
            signatures[p, d] = min_pos
    
    return signatures.astype(int)

# -----------------------
# MinHash via multiple hash functions
# -----------------------

def minhash_hash_functions(doc_shingles_ids, num_perm=50, max_shingle_id=None, seed=42):
    """
    MinHash signature using multiple independent hash functions.
    h_i(x) = (a_i * x + b_i) % prime
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if max_shingle_id is None:
        max_shingle_id = max((sid for s in doc_shingles_ids for sid in s), default=0) + 1

    prime = 2_147_483_647  # large prime
    
    # Generate a_i, b_i
    a = np.random.randint(1, prime-1, size=num_perm, dtype=np.int64)
    b = np.random.randint(0, prime-1, size=num_perm, dtype=np.int64)
    
    n_docs = len(doc_shingles_ids)
    signatures = np.full((num_perm, n_docs), fill_value=np.inf)
    
    for d, sset in enumerate(doc_shingles_ids):
        if not sset:
            continue
        # for each hash function, compute min hash over shingles
        s_ids = np.array(list(sset), dtype=np.int64)
        for i in range(num_perm):
            # vectorized hash
            hash_vals = (a[i] * s_ids + b[i]) % prime
            signatures[i, d] = hash_vals.min()
    
    return signatures.astype(np.int64)

# -----------------------
# Similarity from signatures
# -----------------------

def estimated_jaccard_from_signatures(sig_matrix, i, j):
    """
    Estimate Jaccard similarity between doc i and j from their MinHash signatures.
    sig_matrix: num_perm x num_docs
    """
    sig_i = sig_matrix[:, i]
    sig_j = sig_matrix[:, j]
    return np.mean(sig_i == sig_j)

# -----------------------
# Shingling + MinHash from in-memory texts
# -----------------------

def minhash_from_texts(texts, k=3, num_perm=50, method="hash"):
    """
    texts: list of strings
    method: "hash" or "perm"
    Returns: doc_shingles_ids, signatures
    """
    doc_shingles_ids, shingle_to_id = build_shingle_sets(texts, k=k)
    if method == "perm":
        sigs = minhash_random_permutations(doc_shingles_ids, num_perm=num_perm)
    else:
        sigs = minhash_hash_functions(doc_shingles_ids, num_perm=num_perm,
                                      max_shingle_id=len(shingle_to_id))
    return doc_shingles_ids, shingle_to_id, sigs

# Example usage (NO CSV):
if __name__ == "__main__" and False:  # set to True to run
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumped over a lazy dog",
        "totally different content not similar at all"
    ]
    doc_shingles_ids, shingle_to_id, sigs_hash = minhash_from_texts(docs, k=3, num_perm=100, method="hash")
    sim_0_1 = estimated_jaccard_from_signatures(sigs_hash, 0, 1)
    sim_0_2 = estimated_jaccard_from_signatures(sigs_hash, 0, 2)
    print("Est. Jaccard(0,1) =", sim_0_1)
    print("Est. Jaccard(0,2) =", sim_0_2)

# -----------------------
# Shingling + MinHash from CSV
# -----------------------

def minhash_from_csv(csv_path, text_col="text", k=3, num_perm=50, method="hash"):
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    return minhash_from_texts(texts, k=k, num_perm=num_perm, method=method)

# Example usage (WITH CSV):
if __name__ == "__main__" and False:  # set to True to run
    doc_shingles_ids, shingle_to_id, sigs_perm = minhash_from_csv(
        "docs.csv", text_col="content", k=3, num_perm=50, method="perm"
    )
    print("Signature matrix shape:", sigs_perm.shape)
