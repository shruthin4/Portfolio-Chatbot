import os
import hashlib
import logging
import pdfplumber
import chromadb
import tempfile
import shutil
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nlp_utils import cleaning, extract_words
import google.generativeai as genai


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("chroma_db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is not set")

is_production = os.environ.get('RENDER', False) or os.environ.get('PRODUCTION', False)

if is_production:
    DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_" + str(hash(os.getpid())))
    
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
            logger.info(f"Removed old ChromaDB directory: {DB_DIR}")
        except Exception as e:
            logger.error(f"Error removing old directory: {e}")
else:
    DB_DIR = "./fresh_chroma_db"  

logger.info(f"Using ChromaDB directory: {DB_DIR}")
COLLECTION_NAME = "my_collection"

os.makedirs(DB_DIR, exist_ok=True)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(" Gemini API configured successfully")
else:
    logger.warning("GEMINI_API_KEY not set, embeddings will not work")


def gemini_embedding_function(texts):
    if not GEMINI_API_KEY:
        logger.warning("No API key - returning dummy embeddings")
        return [[0.0] * 768 for _ in range(len(texts))]  
        
    embeddings = []
    for text in texts:
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            embeddings.append([0.0]*768)  # fallback dummy vector
    return embeddings


class GeminiEmbeddings:
    def embed_documents(self, texts):
        return gemini_embedding_function(texts)

    def embed_query(self, text):
        return gemini_embedding_function([text])[0]

logger.info(" Gemini Embedding setup complete!")
embedding_model = GeminiEmbeddings()

vector_db = None

def initialize_chromadb():
    """Initializing ChromaDB and return a reference to it"""
    global vector_db
    
    try:
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        vector_db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model
        )
        logger.info(
            " ChromaDB Initialized successfully in directory: %s", DB_DIR
        )
        
        docs = vector_db.get()
        doc_count = len(docs.get("ids", []))
        logger.info(f"ChromaDB contains {doc_count} documents")
        
        if doc_count == 0:
            # Load documents from the Documents folder
            docs_loaded = store_documents_in_chromadb("./Documents", reset_db=False, use_chunking=False)
            if docs_loaded:
                logger.info(" Documents loaded into ChromaDB")
            else:
                logger.error(" Failed to load documents")
        
        return vector_db
        
    except Exception as e:
        logger.error(f" Error initializing ChromaDB: {e}")

        try:
            if os.path.exists(DB_DIR):
                shutil.rmtree(DB_DIR)
                logger.info(f"Removed problematic ChromaDB directory: {DB_DIR}")
            os.makedirs(DB_DIR, exist_ok=True)
            
            # Try again with a fresh directory
            chroma_client = chromadb.PersistentClient(path=DB_DIR)
            vector_db = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model
            )
            
            store_documents_in_chromadb("./Documents", reset_db=False, use_chunking=False)
            logger.info(" ChromaDB re-initialized successfully after cleanup")
            return vector_db
            
        except Exception as e2:
            logger.error(f" Error reinitializing ChromaDB: {e2}")
            
            logger.warning(" Using in-memory ChromaDB as fallback")
            chroma_client = chromadb.EphemeralClient()
            vector_db = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model
            )
            
            store_documents_in_chromadb("./Documents", reset_db=False, use_chunking=False)
            return vector_db

def read_pdf(file_path):
    text_chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

    return "\n".join(text_chunks)

def load_documents(main_folder):
    all_texts = []
    
    if not os.path.exists(main_folder):
        logger.warning(f"Documents folder '{main_folder}' does not exist")
        current_dir = os.getcwd()
        possible_paths = [
            os.path.join(current_dir, "Documents"),
            os.path.join(current_dir, "..", "Documents"),
            os.path.join(current_dir, "app", "Documents"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Documents folder at: {path}")
                main_folder = path
                break
        else:
            logger.error("Could not find Documents folder")
            return all_texts
    
    logger.info(f"Loading documents from: {main_folder}")
    
    for root, _, files in os.walk(main_folder):
        category = os.path.basename(root)
        for file in files:
            file_path = os.path.join(root, file)
            logger.info(f"Processing file: {file_path}")

            text_content = ""
            if file.lower().endswith(".pdf"):
                text_content = read_pdf(file_path)
            elif file.lower().endswith(".txt") or file.lower().endswith(".md"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    logger.error(f"Error reading text file {file_path}: {e}")
            else:
                logger.info(f"Skipping {file_path} (unsupported extension).")
                continue

            text_content = text_content.strip()
            if not text_content:
                logger.info(f"Skipping {file_path} - extracted text is empty.")
                continue

            all_texts.append({
                "text": text_content,
                "source": file,
                "category": category
            })
    
    logger.info(f"Loaded {len(all_texts)} documents")
    return all_texts

def chunk_text(text, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def store_documents_in_chromadb(main_folder, reset_db=False, use_chunking=False):
    global vector_db
    
    if reset_db:
        logger.info(" Resetting ChromaDB...")
        try:
            if hasattr(vector_db.client, "delete_collection"):
                vector_db.client.delete_collection(COLLECTION_NAME)
            vector_db = Chroma(
                client=vector_db.client,
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_model
            )
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False

    documents = load_documents(main_folder)
    
    if not documents:
        logger.warning("No documents found to add")
        return False
        
    existing_ids = set(vector_db.get().get("ids", []))

    new_texts, new_metadatas, new_ids = [], [], []

    for doc in documents:
        if use_chunking:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(chunk.encode()).hexdigest()
                if doc_id in existing_ids:
                    logger.warning(f" Skipping duplicate chunk: {doc['source']} chunk {i}")
                    continue
                new_texts.append(chunk)
                new_metadatas.append({
                    "source": f"{doc['source']} (chunk {i})",
                    "category": doc["category"]
                })
                new_ids.append(doc_id)
        else:
            doc_id = hashlib.md5(doc["text"].encode()).hexdigest()
            if doc_id in existing_ids:
                logger.warning(f" Skipping duplicate doc {doc['source']}")
                continue
            new_texts.append(doc["text"])
            new_metadatas.append({
                "source": doc["source"],
                "category": doc["category"]
            })
            new_ids.append(doc_id)

    if new_texts:
        try:
            vector_db.add_texts(texts=new_texts, metadatas=new_metadatas, ids=new_ids)
            logger.info(f" Stored {len(new_texts)} new documents.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False
    else:
        logger.warning(" No new documents added.")
        return False

def list_chromadb_documents():
    try:
        stored_docs = vector_db.get()
        if not stored_docs or not stored_docs.get("ids"):
            return []
        return [
            {"id": doc_id, "source": meta.get("source", "Unknown Document"), "category": meta.get("category", "General")}
            for doc_id, meta in zip(stored_docs["ids"], stored_docs["metadatas"])
        ]
    except Exception as e:
        logger.error(f"Error retrieving stored docs: {str(e)}")
        return []

def detect_folder_category(user_message):
    doc_list = list_chromadb_documents()
    categories = set(d["category"].lower() for d in doc_list)
    msg = user_message.lower()
    matched = [cat for cat in categories if cat in msg]
    return matched[0] if matched else None

def query_chromadb(user_query, folder_category=None):
    """
    Query the ChromaDB with enhanced search logic to ensure comprehensive results.
    This improves on the previous version by ensuring both semantic similarity and category-based search.
    """
    logger.info(f"User Query: {user_query}, folder_category: {folder_category}")

    try:
        cleaned_q = cleaning(user_query)
        extracted = extract_words(user_query)
        enhanced_q = cleaned_q + (" " + " ".join(extracted.keys()) if extracted else "")

        all_docs = vector_db.get()
        if not all_docs or not all_docs.get("ids", []):
            logger.warning("No documents in database")
            return []
            
        total_docs = len(all_docs.get("ids", []))
        if total_docs == 0:
            logger.warning("No documents in database to search")
            return []
            
        k_docs = min(total_docs, 15) 
        all_results = vector_db.similarity_search_with_score(enhanced_q, k=k_docs)
        
        similarity_docs = [
            {"source": doc.metadata.get("source", "Unknown"), 
             "category": doc.metadata.get("category", "General"), 
             "text": doc.page_content}
            for doc, _ in all_results
        ]
        
        category_docs = []
        if folder_category:
            try:
                cat_doc_ids = []
                for doc in list_chromadb_documents():
                    if doc["category"].lower() == folder_category.lower():
                        cat_doc_ids.append(doc["id"])
                
                if cat_doc_ids:
                    stored_raw = vector_db.get(ids=cat_doc_ids)
                    for idx, doc_id in enumerate(stored_raw["ids"]):
                        meta = stored_raw["metadatas"][idx]
                        page_content = stored_raw["documents"][idx]
                        category_docs.append({
                            "source": meta.get("source", "Unknown"), 
                            "category": meta.get("category", "General"), 
                            "text": page_content
                        })
            except Exception as e:
                logger.error(f"Error retrieving category documents: {e}")
        
        # IMPORTANT: Always search all documents for skills, projects, etc.
        # These are common keywords that should search everywhere
        important_keywords = ["skill", "project", "experience", "education", "certification", "background"]
        additional_docs = []
        
        if any(keyword in user_query.lower() for keyword in important_keywords):
            logger.info(f"Important keyword detected in query: {user_query}")
            
            try:
                # Get all document IDs that aren't already in our results
                existing_ids = set(doc.metadata.get("source", "") for doc, _ in all_results)
                additional_doc_ids = []
                
                for doc in list_chromadb_documents():
                    if doc["source"] not in existing_ids:
                        additional_doc_ids.append(doc["id"])

                if additional_doc_ids:
                    additional_doc_ids = additional_doc_ids[:20] 
                    add_raw = vector_db.get(ids=additional_doc_ids)
                    
                    for idx, doc_id in enumerate(add_raw["ids"]):
                        meta = add_raw["metadatas"][idx]
                        page_content = add_raw["documents"][idx]
                        additional_docs.append({
                            "source": meta.get("source", "Unknown"), 
                            "category": meta.get("category", "General"), 
                            "text": page_content
                        })
            except Exception as e:
                logger.error(f"Error retrieving additional documents: {e}")
        
        # Combines all results and removes duplicates
        combined_docs = similarity_docs + category_docs + additional_docs
        return unify_docs(combined_docs)
        
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {str(e)}")
        return []

def unify_docs(doc_list):
    unique_map = {}
    for d in doc_list:
        key = (d["source"], d["text"])
        if key not in unique_map:
            unique_map[key] = d
    return list(unique_map.values())

def combine_docs_text(documents):
    raw = [f"[Source: {doc['source']} - Category: {doc['category']}]:\n{doc['text']}" for doc in documents]
    lines = "\n".join(raw).split("\n")
    deduped = []
    seen = set()
    for line in lines:
        norm = line.strip()
        if norm not in seen:
            seen.add(norm)
            deduped.append(line)
    return "\n".join(deduped)


vector_db = initialize_chromadb()


if __name__ == "__main__":
    logger.info("Running ChromaDB setup script directly")
    store_documents_in_chromadb("./Documents", reset_db=True, use_chunking=False)
