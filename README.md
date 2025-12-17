# MedLit Assist 🏥

<p align="center">
  <img src="./public/logo_dark.svg" alt="MedLit Assist Logo" width="200"/>
</p>

An LLM biomedical research assistant that makes medical literature accessible to everyone.

## Features

- 🔍 **Search PubMed Central** for the latest biomedical research
- 📚 **Explain complex research** in simple, easy-to-understand language
- 💡 **Answer questions** about scientific articles
- 🎯 **Synthesize findings** from multiple research papers
- 🗣️ **Translate medical jargon** into everyday language without losing accuracy

## Prerequisites

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

### 2. Download the GPT-OSS:20B Model

After installing Ollama, download the required model:

```bash
ollama pull gpt-oss:20b
```

You will need sufficient space because the model is large (~11GB).

### 3. Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
EMAIL=your.email@example.com
PMC_API_KEY=your_ncbi_api_key_here
```

**Getting your NCBI API Key:**
1. Visit [NCBI API Key Settings](https://www.ncbi.nlm.nih.gov/account/settings/)
2. Sign in or create an account
3. Generate an API key
4. Add it to your `.env` file

> **Note:** While the NCBI E-utilities can work without an API key, having one increases your rate limit from 3 requests/second to 10 requests/second.

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd medlit-assist
```

2. Create and activate a virtual environment with Python 3.12:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Chainlit application:

```bash
chainlit run app.py
```

Or with auto-reload during development:

```bash
chainlit run app.py -w
```

The application will be available at `http://localhost:8000`

### Using the Assistant

Ask questions about biomedical topics:

- "What's the latest research on diabetes treatment?"
- "Tell me about cancer immunotherapy"
- "How does aspirin work in the body?"
- "What do we know about Alzheimer's prevention?"

The assistant will:
1. Search PubMed Central for relevant articles
2. Analyze the research
3. Explain findings in simple, accessible language
4. Provide links to the original articles


## Technologies Used

- **Chainlit** - Interactive chat interface
- **LangChain** - LLM orchestration and tool integration
- **Ollama** - Local LLM inference
- **Biopython** - NCBI E-utilities wrapper with Entrez
- **PubMed Central** - Biomedical literature database

