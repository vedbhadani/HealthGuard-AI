from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import glob

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_index():
    """Build FAISS index for medical knowledge retrieval"""
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load all documents
    docs = []
    metadata = []
    
    print("Loading knowledge base documents...")
    for filepath in glob.glob('rag/knowledge_base/*.txt'):
        print(f"Processing {os.path.basename(filepath)}...")
        with open(filepath, 'r') as f:
            text = f.read()
            chunks = chunk_text(text)
            docs.extend(chunks)
            metadata.extend([{'source': os.path.basename(filepath), 'chunk': i} 
                           for i in range(len(chunks))])
    
    print(f"Total document chunks: {len(docs)}")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save artifacts
    os.makedirs('rag/indexes', exist_ok=True)
    
    print("Saving index and documents...")
    faiss.write_index(index, 'rag/indexes/medical_knowledge.index')
    with open('rag/indexes/documents.pkl', 'wb') as f:
        pickle.dump({'docs': docs, 'metadata': metadata}, f)
    with open('rag/indexes/model_name.txt', 'w') as f:
        f.write('sentence-transformers/all-MiniLM-L6-v2')
    
    print(f"Successfully indexed {len(docs)} document chunks")
    print(f"Index dimension: {dimension}")
    print(f"Index size: {len(docs)} vectors")

if __name__ == '__main__':
    build_index()
