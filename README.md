## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
urls = [
    "https://openreview.net/pdf?id=0wSlFpMsGb",
    "https://openreview.net/pdf?id=MS9nWFY7LG",
    "https://openreview.net/pdf?id=m5byThUSNE",
]

papers = [
    "pdf1.pdf",
    "pdf2.pdf",
    "pdf3.pdf",
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "What are the similarities in pdf1,pdf2 and pdf3 ")
response = agent.query("Compare details of pdf2 and pdf3")
print(str(response))
```

### OUTPUT:
![EXP4 1](https://github.com/user-attachments/assets/1d15e3a3-be6f-4fe8-8b86-9226dc5de453)
![EXP4 2](https://github.com/user-attachments/assets/82b8dff0-1ddc-4af7-ba4a-5059d1add1df)
![EXP4 3](https://github.com/user-attachments/assets/f960e155-cbf5-408e-b675-6b735bf7c7a1)

### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
