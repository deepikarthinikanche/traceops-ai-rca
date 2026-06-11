# TraceOps — AI-Powered Root Cause Analysis System

TraceOps is an AI-powered Root Cause Analysis (RCA) platform designed to automate log analysis and incident investigation using Retrieval-Augmented Generation (RAG). The system helps engineers identify potential root causes of production issues by retrieving relevant logs and generating intelligent troubleshooting insights.

## Features

* Automated log analysis and incident investigation
* Retrieval-Augmented Generation (RAG) workflow
* Vector-based log retrieval using ChromaDB
* FastAPI backend for scalable API services
* Intelligent issue correlation and troubleshooting
* Real-time incident analysis support
* Extensible architecture for observability and monitoring integrations

## Architecture

```text
Logs → ChromaDB Vector Store → Retrieval Layer → LLM/RAG Engine → RCA Response
```

## Tech Stack

### Backend

* FastAPI
* Python

### AI & Retrieval

* LangChain
* ChromaDB
* Sentence Transformers

### Future Integrations

* Grafana
* Prometheus
* NVIDIA DCGM Exporter
* OpenAI / NVIDIA NIM

## Project Structure

```text
traceops-ai-rca/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── sample_logs.txt
│   └── .env
│
└── frontend/
    ├── src/
    ├── public/
    └── package.json
```

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/traceops-ai-rca.git
cd traceops-ai-rca
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Backend

```bash
uvicorn main:app --reload
```

API Documentation:

```text
http://localhost:8000/docs
```

## Example Workflow

### Upload Logs

```http
POST /upload-logs
```

### Analyze Incident

```http
POST /analyze-incident
```

Request:

```json
{
  "question": "Why is inference latency increasing?"
}
```

Response:

```json
{
  "root_cause": "Possible GPU memory saturation due to increased concurrency.",
  "suggested_fix": "Reduce concurrency and optimize batch size."
}
```

## Future Enhancements

* Multi-agent RCA workflows
* Real-time log ingestion
* Grafana dashboard integration
* Prometheus metrics analysis
* NVIDIA GPU telemetry integration
* Incident timeline generation
* Slack and Jira integration
* Kubernetes observability support

## Use Cases

* Production incident investigation
* AI infrastructure monitoring
* Application performance troubleshooting
* LLM inference debugging
* Enterprise observability workflows

## Author

Deepikarthini Kanche

## License

This project is licensed under the MIT License.
