# Build-a-RAG-ChatBot-related-to-Medical-
Build a RAG ChatBot related to Medical and frontend build with Flask and using LLM, Langchain and PineCone(Vector database) 
---

## 🧰 Tech Stack

- **Python 3.10+** — core language  
- **Flask** — web server / frontend  
- **LangChain** — orchestrating LLM + retrieval + chaining  
- **OpenAI / Other LLM providers** — underlying language model for generation  
- **Pinecone** — vector database for embeddings and similarity search  
- **python-dotenv** — environment variable management  

---

## Key Components

agents/: Defines various AI agents responsible for specific tasks within the medical chatbot framework.

tasks/: Outlines the tasks that agents will execute, detailing the objectives and workflows.

tools/: Contains utility functions and tools that support the agents and tasks.

app.py: Serves as the main application entry point, initializing and managing the execution of agents and tasks.

crew.py: Coordinates the AI agents, ensuring they work together effectively.

main.py: Launches the application, setting up necessary configurations and starting the agent workflows.

requirements.txt: Lists all the Python dependencies required to run the project.

README.md: Provides an overview of the project, setup instructions, and usage guidelines.
---
## Core Components

agents/: Contains definitions of various AI agents responsible for different tasks within the medical chatbot framework.

tasks/: Houses the task definitions that agents will execute, outlining specific objectives and workflows.

tools/: Includes utility functions and tools that support the agents and tasks.

app.py: Acts as the main application entry

---

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
---

