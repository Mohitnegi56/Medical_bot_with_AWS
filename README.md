# Medical Chatbot - Retrieval-Augmented Generation (RAG)

A modern, responsive, and lightweight AI-powered Medical Chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline to answer medical queries. It leverages **LangChain**, **Groq** (Llama-3.3-70b), **Pinecone** (Vector Database), and **Hugging Face Serverless Inference API** (for embeddings) to provide highly accurate, context-aware answers from medical literature.

### Live Demo
🌐 **Render App**: [https://medical-bot-with-aws.onrender.com/](https://medical-bot-with-aws.onrender.com/)

---

## 🚀 Key Features

*   **RAG Pipeline**: Retrieves contextually relevant documents from a Pinecone vector index containing medical literature (PDFs) to ground responses.
*   **Hosted Embeddings**: Uses Hugging Face's Serverless Inference API (`all-MiniLM-L6-v2`) for computing text embeddings, reducing memory requirements to less than **50 MB** (perfect for free-tier cloud environments).
*   **Modern UI**: Built with a sleek, responsive, and interactive interface featuring:
    *   Responsive card grid with clickable suggestion prompts.
    *   Bouncing typing indicator.
    *   Dynamic chat bubble transitions and smooth scroll.
    *   Optimized layout for both desktop and mobile viewports.
*   **Production Deployment Ready**: Preconfigured for both AWS (CI/CD) and Render PaaS.

---

## 🛠️ Technology Stack

*   **Frontend**: HTML5, CSS3 (Glassmorphism, custom scrollbars, animations), Vanilla JavaScript.
*   **Backend**: Flask (Python).
*   **RAG Orchestration**: LangChain, `langchain-groq`, `langchain-pinecone`, `langchain-huggingface`.
*   **Database**: Pinecone Serverless Vector Store.
*   **LLM Model**: Groq Cloud - `llama-3.3-70b-versatile`.
*   **Embeddings Model**: Hugging Face Inference Hub - `sentence-transformers/all-MiniLM-L6-v2`.

---

## 🔧 Environment Variables

The application requires the following environment variables to be set (either in a local `.env` file or in your cloud provider's dashboard):

```ini
PINECONE_API_KEY="your-pinecone-api-key"
GROQ_API_KEY="your-groq-api-key"
HF_TOKEN="your-huggingface-read-token"
```

*Note: Generating a free Hugging Face User Access Token (`HF_TOKEN`) is highly recommended to authenticate requests to the Inference API and avoid rate limits.*

---

## 🚢 Deployment Architecture

This project is configured for two deployment models:

### 1. AWS CI/CD Pipeline (Docker & GitHub Actions)
The repository contains a fully automated GitHub Actions pipeline (`.github/workflows/cicd.yaml`) configured to deploy the container to an **AWS EC2 instance**:
*   **Dockerfile**: Containerizes the Flask application utilizing a lightweight `python:3.10-slim-buster` base.
*   **Continuous Integration (CI)**:
    1. Triggers on pushes to the `main` branch.
    2. Builds the Docker image.
    3. Authenticates with AWS credentials.
    4. Pushes the container image to **Amazon Elastic Container Registry (ECR)**.
*   **Continuous Deployment (CD)**:
    1. Triggers a self-hosted runner hosted on your **AWS EC2** instance.
    2. Logs into Amazon ECR.
    3. Pulls the latest Docker image.
    4. Automatically stops and removes any existing container named `medical-bot` to prevent port conflicts.
    5. Spins up the new container mapped to port `8000` with the necessary secrets loaded.

### 2. Render Cloud Deployment
For environments with strict resource constraints (e.g., Render Free Tier with a **512 MB RAM limit**):
*   By migrating from local CPU embeddings (which load heavy PyTorch/Transformer weights in memory) to Hugging Face's hosted Serverless API (`HuggingFaceEndpointEmbeddings`), we removed the dependency on PyTorch and `sentence-transformers`.
*   As a result, memory consumption was reduced from **>1.2 GB to less than 50 MB**, allowing the app to run smoothly on Render's free tier.
*   **Render Port Detection**: The web server is configured to dynamically bind to the `PORT` environment variable assigned by Render:
    ```python
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
    ```

---

## 💻 Local Setup and Run

### 1. Clone the Repository
```bash
git clone https://github.com/Mohitnegi56/Medical_bot_with_AWS.git
cd Medical_bot_with_AWS
```

### 2. Setup Virtual Environment & Dependencies
```bash
python -m venv myenv
# On Windows
.\myenv\Scripts\activate
# On Linux/macOS
source myenv/bin/activate

pip install -r requirements.txt
```

### 3. Setup Secrets
Create a `.env` file in the root directory and add your keys:
```ini
PINECONE_API_KEY="your_pinecone_api_key"
GROQ_API_KEY="your_groq_api_key"
HF_TOKEN="your_huggingface_token"
```

### 4. Vector Store Ingestion (Optional)
If you need to populate or update the Pinecone index with your own PDF documents under `data/`:
```bash
python store_index.py
```

### 5. Run the Application
```bash
python app.py
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.