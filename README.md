# 🚀 AI Projects Portfolio

A professional, engineering-focused repository showcasing **end-to-end AI systems**, combining LLMs, real-world problem solving, evaluation frameworks, and production-grade architecture.

This repository reflects a transition from traditional Machine Learning to **applied AI systems & intelligent agents**, with a strong focus on:

- Real-world AI use cases  
- Scalable and modular architecture  
- LLM-powered pipelines  
- Evaluation & reliability of AI systems  
- Clean engineering and API-first design  

---

## 🧠 Featured Project

### 📄 Invoice Document Processing Agent

An **AI-powered invoice processing pipeline** that extracts structured data from PDF invoices using LLMs, validates outputs, and evaluates performance using hybrid scoring techniques.

#### 🔍 Key Capabilities

- 📥 PDF → Text extraction using `pdfplumber`  
- 🤖 Structured field extraction using Azure OpenAI  
- ✅ Validation against expected values using LLM reasoning  
- 📊 Dual evaluation system:
  - Deterministic scoring (field-level accuracy)
  - LLM-as-Judge (holistic quality scoring)

---

## 🏗 Architecture Overview

The system is designed as a modular pipeline exposed via APIs:

- **FastAPI Server**
  - `/api/process_invoice`
  - `/api/evaluate`
  - `/health`

### Processing Flow:

1. PDF Extraction  
2. LLM Field Extraction  
3. LLM Validation  
4. Evaluation Layer  

---

## ⚙️ Tech Stack

- **Backend:** FastAPI, Uvicorn  
- **LLM:** Azure OpenAI (Managed Identity)  
- **PDF Processing:** pdfplumber  
- **Config Management:** Pydantic Settings  
- **Evaluation:** Custom deterministic + LLM judge  
- **Testing:** Pytest (fully mocked, no cloud dependency)  
- **Logging:** Structured JSON logging  

---

## 📂 Repository Structure

```
AI_Projects/
│
├── Invoice_Document_Processing_Agent/
│   ├── src/
│   ├── tests/
│   ├── sample_invoices/
│   ├── main.py
│   └── README.md
│
└── (Upcoming Projects)
```

---

## 🧪 Engineering Highlights

- 🔹 Modular pipeline design (orchestrator-driven)  
- 🔹 Strong separation of concerns (pipeline, models, evaluation)  
- 🔹 Production-ready API layer  
- 🔹 Managed identity authentication (no hardcoded secrets)  
- 🔹 Evaluation-first approach (critical for AI reliability)  
- 🔹 Fully testable system with mocked dependencies  

---

## 🚀 Roadmap

This repository will expand into advanced AI system design:

- 🧾 Multi-document understanding agents  
- 🧠 Retrieval-Augmented Generation (RAG) systems  
- 🤖 Autonomous AI agents & workflows  
- 📊 AI evaluation frameworks & benchmarking  
- 🔐 Responsible AI & governance patterns  

---

## 🎯 Objective

This repository demonstrates:

✔ Applied AI system design  
✔ Strong engineering discipline  
✔ Real-world problem solving using LLMs  
✔ Focus on evaluation, reliability, and scalability  
✔ Continuous evolution in AI capabilities  

---

## 👤 Author

**Vinay Sangam**  
Data & AI Engineer  

---

## ⭐ Support

If you find this repository useful:

- ⭐ Star the repo  
- 🍴 Fork and explore  
- 🤝 Connect and collaborate  
