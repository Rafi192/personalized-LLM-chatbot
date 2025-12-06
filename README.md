# personalized-LLM-chatbot



## 📝 Description

Develop a personalized LLM chatbot using cutting-edge technology. This project provides a robust API for seamless integration, aa reliable database for efficient data management, and thorough testing to ensure optimal performance and a superior user experience. Unleash the power of customized AI interactions with this comprehensive solution.

##  Features
-  **Multi-Collection RAG** - Intelligent retrieval from multiple MongoDB collections
-  **MongoDB Atlas Integration** - Cloud-based data storage
-  **Advanced Embeddings** - BGE-M3 multilingual embeddings with FAISS vector search
-  **Gemini AI** - Powered by Google's latest Gemini model
-  **Smart Retrieval** - Collection-specific filtering and relevance scoring
-  **Comprehensive Testing** - Full test suite for validation

### Prerequisites

- Python 3.11 or greater
- requirements.txt
- Google Gemini API key
- Git

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/your-username/personalized-LLM-chatbot.git
   cd personalized-LLM-chatbot
```

2. **Create virtual environment**
```bash
   # Windows
   python -3.12 -m venv rag_chatbot_env
   rag_chatbot_env\Scripts\activate

   # Linux/Mac
   python3.12 -m venv rag_chatbot_env
   source rag_chatbot_env/bin/activate
   ```
3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
```env
   # MongoDB Configuration
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net
   MONGODB_DATABASE=test

   # Google Gemini API
   GEMINI_API_KEY=your_gemini_api_key_here
```
5. **Run Api**

```env
   uvicorn main:app --reload
```
## 💻 Running Locally

### Step 1: Verify Database Connection
```bash
cd src/db
python test_db.py
```

Expected output:
```
databases ['test', 'admin', 'local']
Test database collections : ['doctors', 'faqs', 'treatmentlists', ...]
```

### Step 2: Run Data Ingestion (One-Time Setup)

This creates the FAISS vector index from your MongoDB collections:
```bash
cd src/ingestion
python ingest_multi_collection_mongodb.py
```

This will:
- ✅ Connect to MongoDB Atlas
- ✅ Load documents from multiple collections
- ✅ Generate embeddings using BGE-M3
- ✅ Build FAISS index
- ✅ Save to `data/embeddings/medical_practice_vectors/`

Expected output:
```
🏥 MULTI-COLLECTION MEDICAL DATABASE RAG INGESTION
======================================================================
[1] Configuration:
    📊 Database: test
    🧠 Embedding Model: BAAI/bge-m3
    💾 Vector Store: data/embeddings/medical_practice_vectors

...

✅ INGESTION COMPLETED SUCCESSFULLY!
📊 Summary:
   - Total Documents: 156
   - Collections Processed: 8
   - Embedding Dimension: 1024
```

### Step 3: Test the Chatbot
```bash
cd tests
python new_test_mongo.py
```

**Interactive testing:**
```
Medical Practice Chatbot - Type 'exit' to quit
======================================================================

You: Who are the orthodontists?

[Retrieved 3 documents]
  1. doctors (score: 0.856)
  2. doctors (score: 0.782)
  3. faqs (score: 0.654)

AI: We have two Specialist Orthodontists:
- Dr. James Herman: Australian-born specialist...
- Dr. Anca Herman: Highly skilled specialist...

----------------------------------------------------------------------
 You: exit
👋 Goodbye!
```

Postman Documentation : https://documenter.getpostman.com/view/49369352/2sB3dMyXJZ

## 📁 Project Structure

```
personalized-LLM-chatbot
├── LICENSE
├── data
│   └── embeddings
│       └── medical_practice_vectors
│           ├── documents.pkl
│           └── faiss_index.bin
├── src
│   ├── __init__.py
│   ├── api
│   │   ├── app.py
│   │   └── routes.py
│   ├── db
│   │   ├── test_db.py
│   │   └── test_db_2.py
│   ├── ingestion
│   │   ├── ingest_multi_collection_mongodb.py
│   │   ├── mongodb_indexer.py
│   │   ├── multi_collection_embedder.py
│   │   └── multi_collection_mongodb_loader.py
│   ├── llm
│   │   ├── __init__.py
│   │   ├── augmented_prompt.py
│   │   ├── enhanced_augmented_prompt.py
│   │   └── enhanced_generator.py
│   ├── retriever
│   │   ├── enhanced_mongodb_retriever.py
│   │   └── mongodb_retriever.py
│   └── utils
│       ├── config.py
│       ├── helpers.py
│       └── logger.py
└── tests
    ├── new_test_mongo.py
    └── test_mongo.py
```

Please ensure your code follows the project's style guidelines and includes tests where applicable.


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/your-username/repo.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request
