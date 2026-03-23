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

### 2. Download the Qwen3:8b Model

After installing Ollama, download the required model:

```bash
ollama pull qwen3:8b
```

You will need sufficient space because the model is large (~5GB).

### 3. Clone the repo set virtual environment

1. Clone the repository:
```bash
git clone https://github.com/nickhoulahan/medlit-assist.git
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

### 4. Environment Variables

Create a `.env` file in the root directory.

Copy .env.example to .env

```bash
cp .env.exampl .env
```
Follow the instructions in the report to insert the right varibles.

**Getting your NCBI API Key:**
1. Visit [NCBI API Key Settings](https://www.ncbi.nlm.nih.gov/account/settings/)
2. Sign in or create an account
3. Generate an API key
4. Add your email and the API key to your `.env` file

> **Note:** While the NCBI E-utilities can work without an API key, having one increases your rate limit from 3 requests/second to 10 requests/second.


## Usage

### Running the Application

Ensure that the qwen3:8b model is running by opening a terminal and running:

```bash
ollama run qwen3:8b
```

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

## Tests

Make sure your virtual environment has the requirements-dev.txt dependencies installed. While .venv is activated, run:
```bash
pip install -r requirements-dev.text
```

### Unit test

```bash
python -m pytest tests/unit/
```

Unit test coverage HTML report

```bash
python -m pytest tests/unit/ --cov=src --cov-report=html
```

### Integration tests

```bash
python -m pytest tests/integration/
```

## Evaluations

These require the dev dependencies installed if you want to run them. Note that evaluation reports and jupyter notebooks are
already available in each directory if you want to don't want to run the evals, but look at the reports and analysis.

### RAGAS evals

You can generate a report for a subset of MedQA questions using:

- [Answer Accuracy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/nvidia_metrics/#answer-accuracy)
- [Answer Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)
- [Context Precision](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/)
- [Noise Sensitivity](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/)
- [Semantic Similarity](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/semantic_similarity/)

Test answer relevancy (no reference text available, but uses prepared questions at tests/evals/ragas/doc_questions/)

```bash
python -m tests.evals.ragas.run_medqa_ragas_eval --eval-mode fulltext --agent-model qwen3:8b --evaluator-model gpt-4.1-mini-2025-04-14
```

Test all metrics with random sample of MedQA answer (can vary sample-size and seed to mix results)
Note, keep sample-size small due to duration of tests and tokens needed.
Currently one needs an API key for openAI's GPT4.1 model used in the evaluations (copy into your .env file from report instructions)

```bash
python -m tests.evals.ragas.run_medqa_ragas_eval --eval-mode search --sample-size 3 --seed 42 --agent-model qwen3:8b --evaluator-model gpt-4.1-mini-2025-04-14
```

<details>
  <summary>Open Jupyter notebook</summary>

  [RAGAS analysis notebook](tests/evals/ragas/ragas_analysis.ipynb)
</details>

### ASR evals

```bash
python -m tests.evals.asr.run_asr_evals
```
<details>
  <summary>Open Jupyter notebook</summary>

  [ASR metrics analysis notebook](tests/evals/asr/asr_metrics_analysis.ipynb)
</details>

### TTS evals

```bash
python -m tests.evals.tts.run_tts_evals
```
<details>
  <summary>Open Jupyter notebook</summary>

  [TTS analysis notebook](tests/evals/tts/tts_analysis.ipynb)
</details>


## Technologies Used

- **Chainlit** - Interactive chat interface
- **LangChain** - LLM orchestration and tool integration
- **Ollama** - Local LLM inference
- **Biopython** - NCBI E-utilities wrapper with Entrez
- **PubMed Central** - Biomedical literature database

