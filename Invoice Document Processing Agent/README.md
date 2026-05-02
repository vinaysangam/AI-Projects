# Invoice Document Processing Agent

AI-powered invoice processing pipeline that extracts structured data from PDF invoices using Azure OpenAI, validates results against ground truth, and evaluates accuracy with dual scoring (deterministic + LLM-as-Judge).

**Author:** Vinay Sangam

---

## Architecture

```
+--------------------------------------------------------------+
|                       FastAPI Server                          |
|              POST /api/process_invoice                        |
|              POST /api/evaluate                               |
|              GET  /health                                     |
+---------------------------+----------------------------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
 +-------------+   +---------------+   +---------------+
 |  Step 1:    |   |  Step 2:      |   |  Step 3:      |
 |  PDF Text   |   |  LLM Field   |   |  LLM          |
 |  Extraction |   |  Extraction   |   |  Validation   |
 |             |   |               |   |               |
 |  pdfplumber |   |  Azure OpenAI |   |  Azure OpenAI |
 +------+------+   +-------+-------+   +-------+-------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                   +-----------------+
                   |   Evaluation    |
                   |                 |
                   | - Deterministic |
                   |   (field-by-    |
                   |    field)       |
                   | - LLM-as-Judge  |
                   |   (holistic     |
                   |    0-1 score)   |
                   +-----------------+
```

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/pipeline/pdf_extractor.py` | Extracts raw text from PDF using pdfplumber |
| 2 | `src/pipeline/field_extractor.py` | Sends raw text to Azure OpenAI to extract structured invoice fields |
| 3 | `src/pipeline/validator.py` | Compares extracted fields against expected values via LLM |

## Evaluation

| Method | Module | Description |
|--------|--------|-------------|
| Deterministic | `src/evaluation/deterministic.py` | Field-by-field comparison (float tolerance 0.01, case-insensitive strings) |
| LLM-as-Judge | `src/evaluation/llm_judge.py` | Holistic quality score (0-1) with reasoning |

---

## Project Structure

```
|-- main.py                     # CLI entry point (--serve)
|-- src/
|   |-- app.py                  # FastAPI application
|   |-- config/
|   |   |-- settings.py         # Pydantic-settings configuration
|   |   |-- prompts.py          # LLM prompt templates
|   |   |-- logging_config.py   # Structured JSON logging
|   |-- models/
|   |   |-- patient_vitals.py   # Invoice data models (ExtractedInvoice, ValidationReport)
|   |   |-- alert_models.py     # PipelineState shared state
|   |   |-- evaluation.py       # Evaluation result models
|   |-- pipeline/
|   |   |-- pdf_extractor.py    # Step 1: PDF to raw text
|   |   |-- field_extractor.py  # Step 2: raw text to structured fields
|   |   |-- validator.py        # Step 3: extracted vs expected
|   |   |-- orchestrator.py     # Chains steps 1, 2, 3
|   |-- evaluation/
|   |   |-- deterministic.py    # Field-by-field scorer
|   |   |-- llm_judge.py        # LLM holistic scorer
|   |-- utils/
|       |-- llm_client.py       # Azure OpenAI client (managed identity)
|       |-- helpers.py          # JSON parsing utilities
|       |-- logger.py           # Logging accessor
|-- tests/                      # 59 unit tests
|-- sample_invoices/            # Place PDF invoices here
|-- test_cases.json             # Evaluation dataset
|-- requirements.txt
|-- .env.example
|-- .gitignore
```

---

## Prerequisites

- Python 3.12+
- Azure OpenAI resource with a model deployment
- Azure CLI logged in (`az login`) for managed identity authentication
- RBAC: "Cognitive Services OpenAI User" role assigned to your identity on the Azure OpenAI resource

## Setup

1. Clone the repository

   ```bash
   git clone <repo-url>
   cd "AI Apprentice Capstone Project"
   ```

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Azure OpenAI endpoint and deployment name.

4. Authenticate

   ```bash
   az login
   ```

## Running the Server

```bash
python main.py --serve
```

The server starts at `http://127.0.0.1:8000`. Swagger UI is available at `/docs`.

## API Endpoints

### GET /health

Returns service health status.

### POST /api/process_invoice

Process a single invoice through the full pipeline.

Request:

```json
{
  "pdf_path": "sample_invoices/1000+ PDF_Invoice_Folder/invoice_example.pdf",
  "expected": {
    "invoice_number": "INV-001",
    "total_amount": 1350.00
  }
}
```

Response: Extracted invoice fields, raw text, and validation report.

### POST /api/evaluate

Run batch evaluation on a dataset of test cases.

Request:

```json
{
  "dataset_path": "test_cases.json"
}
```

Response: Per-case results with deterministic and LLM judge scores, plus aggregate averages.

## Running Tests

```bash
python -m pytest tests/ -v
```

All 59 tests pass without requiring Azure credentials (mocked).

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI + Uvicorn |
| PDF Extraction | pdfplumber |
| LLM Provider | Azure OpenAI (via managed identity) |
| Authentication | azure-identity (DefaultAzureCredential) |
| Configuration | pydantic-settings |
| Retry Logic | tenacity |
| Logging | python-json-logger |
| Testing | pytest |