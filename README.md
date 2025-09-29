# Build-a-RAG-ChatBot-related-to-Medical-
Build a RAG ChatBot related to Medical and frontend build with Flask and using LLM, Langchain and PineCone(Vector database) 


# To create Folder structure
```bash
python template.py
```

### Create Conda virtual Environment
```bash
conda create -n medicalchatbot python=3.10 -y
```

### Activate MedicalChatbot
```bash
conda activate medicalchatbot
```
---
### Install the requirements
```bash
pip install -r requirements.txt
```
### Environment Variables

Create a .env file in the root directory:
```bash
PINECONE_API_KEY=your-pinecone-key
GROQ_API_KEY=your-groq-key
```
---
## 🚀 Running the App
```bash
uvicorn app:app --reload
```


