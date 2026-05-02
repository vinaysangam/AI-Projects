# Invoice Document Processing Agent

AI-powered invoice processing pipeline that extracts structured data from PDF invoices using Azure OpenAI, validates results against ground truth, and evaluates accuracy with dual scoring (deterministic + LLM-as-Judge).

**Author:** Vinay Sangam

---

## Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                       FastAPI Server                          â”‚
â”‚              POST /api/process_invoice                        â”‚
â”‚              POST /api/evaluate                               â”‚
â”‚              GET  /health                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â”‚
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â–¼                   â–¼                   â–¼
 â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
 â”‚  Step 1:    â”‚   â”‚  Step 2:      â”‚   â”‚  Step 3:      â”‚
 â”‚  PDF Text   â”‚   â”‚  LLM Field   â”‚   â”‚  LLM          â”‚
 â”‚  Extraction â”‚   â”‚  Extraction   â”‚   â”‚  Validation   â”‚
 â”‚             â”‚   â”‚               â”‚   â”‚               â”‚
 â”‚  pdfplumber â”‚   â”‚  Azure OpenAI â”‚   â”‚  Azure OpenAI â”‚
 â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜   â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
        â”‚                   â”‚                   â”‚
        â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                            â–¼
                   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                   â”‚   Evaluation    â”‚
                   â”‚                 â”‚
                   â”‚ â€¢ Deterministic â”‚
                   â”‚   (field-by-    â”‚
                   â”‚    field)       â”‚
                   â”‚ â€¢ LLM-as-Judge  â”‚
                   â”‚   (holistic     â”‚
                   â”‚    0â€“1 score)   â”‚
                   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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
| LLM-as-Judge | `src/evaluation/llm_judge.py` | Holistic quality score (0â€“1) with reasoning |

---

## Project Structure

```
â”œâ”€â”€ main.py                     # CLI entry point (--serve)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ app.py                  # FastAPI application
â”‚   â”œâ”€â”€ config/
â”‚   â”‚   â”œâ”€â”€ settings.py         # Pydantic-settings configuration
â”‚   â”‚   â”œâ”€â”€ prompts.py          # LLM prompt templates
â”‚   â”‚   â””â”€â”€ logging_config.py   # Structured JSON logging
â”‚   â”œâ”€â”€ models/
â”‚   â”‚   â”œâ”€â”€ patient_vitals.py   # Invoice data models (ExtractedInvoice, ValidationReport)
â”‚   â”‚   â”œâ”€â”€ alert_models.py     # PipelineState shared state
â”‚   â”‚   â””â”€â”€ evaluation.py       # Evaluation result models
â”‚   â”œâ”€â”€ pipeline/
â”‚   â”‚   â”œâ”€â”€ pdf_extractor.py    # Step 1: PDF â†’ raw text
â”‚   â”‚   â”œâ”€â”€ field_extractor.py  # Step 2: raw text â†’ structured fields
â”‚   â”‚   â”œâ”€â”€ validator.py        # Step 3: extracted vs expected
â”‚   â”‚   â””â”€â”€ orchestrator.py     # Chains steps 1â†’2â†’3
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â”œâ”€â”€ deterministic.py    # Field-by-field scorer
â”‚   â”‚   â””â”€â”€ llm_judge.py        # LLM holistic scorer
â”‚   â””â”€â”€ utils/
â”‚       â”œâ”€â”€ llm_client.py       # Azure OpenAI client (managed identity)
â”‚       â”œâ”€â”€ helpers.py          # JSON parsing utilities
â”‚       â””â”€â”€ logger.py           # Logging accessor
â”œâ”€â”€ tests/                      # 59 unit tests
â”œâ”€â”€ sample_invoices/            # Place PDF invoices here
â”œâ”€â”€ test_cases.json             # Evaluation dataset
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .env.example
â””â”€â”€ .gitignore
```

---

## Prerequisites

- Python 3.12+
- Azure OpenAI resource with a model deployment
- Azure CLI logged in (`az login`) for managed identity authentication
- **RBAC:** "Cognitive Services OpenAI User" role assigned to your identity on the Azure OpenAI resource

## Setup

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd "AI Apprentice Capstone Project"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Azure OpenAI endpoint and deployment name.

4. **Authenticate**

   ```bash
   az login
   ```

## Running the Server

```bash
python main.py --serve
```

The server starts at **http://127.0.0.1:8000**. Swagger UI is available at `/docs`.

## API Endpoints

### `GET /health`

Returns service health status.

### `POST /api/process_invoice`

Process a single invoice through the full pipeline.

**Request:**
```json
{
  "pdf_path": "sample_invoices/1000+ PDF_Invoice_Folder/invoice_example.pdf",
  "expected": {
    "invoice_number": "INV-001",
    "total_amount": 1350.00
  }
}
```

**Response:** Extracted invoice fields, raw text, and validation report.

### `POST /api/evaluate`

Run batch evaluation on a dataset of test cases.

**Request:**
```json
{
  "dataset_path": "test_cases.json"
}
```

**Response:** Per-case results with deterministic and LLM judge scores, plus aggregate averages.

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

---

## License

MIT
