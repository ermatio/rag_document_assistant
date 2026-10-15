# RAG-Based AI Assistant

**RAG-Based AI Assistant** is a Python project that answers users' questions using information from PDF documents. The application searches the document for relevant text passages and sends them to **Gemini** to generate answers grounded in those sources.

> **Project status:** Learning project. The current version is designed to run locally. To run the code hosted on GitHub, you will need the required Python libraries, a Gemini API key, and a document index.

## Features

- Extract text from PDF documents.
- Split text into smaller pieces (*chunks*).
- Create vector representations (*embeddings*) of text using `sentence-transformers`.
- Find the passages most relevant to a user's question through semantic search.
- Generate answers with Gemini based on the retrieved information.
- Provide source references or page numbers used in answers.
- Interact with the assistant through a simple Streamlit web interface.

## How Does RAG Work?

**RAG (Retrieval-Augmented Generation)** is an approach in which a system retrieves relevant information from an external source before generating an answer.

The project has two main stages:

**1. Document preparation**

```text
PDF document
    ↓
Text extraction
    ↓
Text chunking
    ↓
Embedding generation
    ↓
Saving embeddings and text chunks
```

**2. Question answering**

```text
User question
    ↓
Create an embedding for the question
    ↓
Search for similar text chunks
    ↓
Send retrieved passages to Gemini
    ↓
Display an answer grounded in the sources
```

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Streamlit | User interface |
| Google Gemini API | Answer generation |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Generating document and question embeddings |
| NumPy | Working with vectors and calculating similarity |
| PyMuPDF | Extracting text from PDFs |
| python-dotenv | Loading environment variables (when needed) |

## Project Structure

```text
RAG-project/
├── app.py                # Streamlit application
├── data/
│   └── extract.py       # PDF text processing
├── index/
│   └── search.py        # Semantic search logic
├── .gitignore            # Files excluded from Git tracking
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

**Local files not uploaded to GitHub:** `data/lecture.pdf`, `index/embeddings.npy`, and `index/chunks.json`. These may be needed when processing documents. The PDF was not included because it may be protected by copyright.

## Installation and Usage

### 1. Download the Repository

Download it from your GitHub repository page using **Code → Download ZIP**, or clone it using your own repository URL:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Your Gemini API Key

Set your own Gemini API key as an environment variable in Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

This command sets the key for the current terminal session. If your code uses a different environment variable name, change the command to match it.

### 5. Prepare the PDF and Document Index

Place a PDF that you have the right to use in the location expected by the project, such as `data/lecture.pdf`. Then run the document preparation script:

```powershell
python data/extract.py
```

Depending on the current implementation of the script, additional configuration or indexing steps may be required. Before launching the application, make sure the necessary index files, such as `index/embeddings.npy` and `index/chunks.json`, are present.

### 6. Run the Application

```powershell
streamlit run app.py
```

Streamlit will display a local web address in the terminal. Open that address in your browser to ask questions about the document.

## Important Notes

- AI-generated answers can be incorrect. Verify important information against the original document.
- Answer quality depends on how well text is extracted, chunked, and retrieved.
- The code on GitHub alone is not sufficient: you must configure the PDF, document index, and API key in your local environment.
- Do not upload API keys, virtual environment files, or copyright-protected documents to a public repository.

## Project Goal

The goal of this project is to gain practical experience with **RAG systems**, **embeddings**, **semantic search**, **LLM APIs**, and **building AI applications in Python**.