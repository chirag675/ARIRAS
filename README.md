# ARIRAS - AI Regulatory Intelligence & Reporting Assurance System
---
HOSTED ON STREAMLIT 
## What is ARIRAS?

ARIRAS is a multi-agent AI system that helps Indian and Global enterprises understand, interpret, and comply with government regulations - without needing a team of lawyers or a ₹50 lakh consulting budget.

**The problem it solves:**

Let's start from Indian Context
- 81% of Indian enterprises have not updated their DPDP-aligned privacy policies
- 83% haven't started end-to-end implementation
- 71% say they simply cannot interpret what regulations require
- Maximum fine under DPDP Act: ₹250 crore per violation

If we go a global - 
- Compliance complexity doesn’t scale linearly -  it explodes. Enterprises today deal with overlapping regulations like GDPR, SOX, HIPAA, and Basel III
- Each regulation brings its own structure, language, and interpretation challenges
- Most organizations operate across jurisdictions - multiplying compliance complexity
- Despite heavy spending on compliance:
    - Companies still rely on manual interpretation and static policies
    - Compliance remains reactive, fragmented, and expensive
The result: not just penalties - but delayed decisions, operational inefficiencies, and hidden risk exposure at scale.

Even the most advanced tech companies struggle with compliance at scale. Google has faced multiple regulatory penalties under GDPR
In 2019, it was fined €50 million by the French regulator for lack of transparency and valid consent

**What ARIRAS does:**

Upload any regulation PDF (DPDP Act, SEBI circular, RBI guideline, Companies Act — anything). ARIRAS reads it, understands it, and gives your company:

1. **Answers to compliance questions** with exact clause citations
2. **Gap analysis** - maps your company policy against the regulation and tells you exactly what's missing
3. **Policy guidance** - asks plain-English questions about your business and generates a tailored compliance guidance report with sample clauses (downloadable as Excel)
4. **Full audit trail** - every decision logged and exportable for regulatory traceability
5. **Compliance dashboard** - live metrics, severity breakdown, score over time

---

## Demo Flow

```
**Feature 1 **
 → Upload DPDP Act / SEBI circular PDF → Index it (30 seconds)
 → Ask: "What are our reporting obligations?" → Get answer with clause citations
**Feature 2 **
 → Upload your company policy → Run gap analysis → Get compliance score + gap list
**Feature 3 **
 → Answer 3 questions about your business → Download policy guidance Excel
**Feature 4 **
 → View dashboard → Export compliance report
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM |Groq - llama-3.3-70b-versatile
 (free, fast) |
| Orchestration | LangChain |
| Vector Database | ChromaDB (local, free) |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 (local, free) |
| PDF Parsing | PyPDF |
| Excel Export | openpyxl |
| Environment | Python 3.10+ |

**Zero cloud cost.** Everything runs locally except the Groq API (free tier).

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                          │
│   Tab 1: Q&A  │  Tab 2: Gap Detector  │  Tab 3: Builder │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│              LANGCHAIN ORCHESTRATION                     │
│         Agent routing · Error handling · Logging        │
└──────┬─────────────────┬──────────────────┬─────────────┘
       │                 │                  │
┌──────▼──────┐  ┌───────▼──────┐  ┌───────▼──────────┐
│  RAG Agent  │  │  Gap Detector│  │  Policy Builder  │
│  rag_agent  │  │  gap_detector│  │  policy_builder  │
└──────┬──────┘  └───────┬──────┘  └───────┬──────────┘
       │                 │                  │
┌──────▼─────────────────▼──────────────────▼──────────┐
│                 RAG INTELLIGENCE LAYER                 │
│   ChromaDB  ←→  HuggingFace Embeddings  ←→  PyPDF    │
│              Regulation PDFs vectorized here          │
└──────────────────────────┬────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Groq — Llama 3.1 70B  │
              │   LLM inference layer   │
              └─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │      AUDIT TRAIL        │
              │  Every decision logged  │
              │  Exportable as JSON     │
              └─────────────────────────┘
```

---

## Project Structure

```
ariras/
│
├── app.py                     # Main Streamlit app — all UI views
│
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py           # Tab 1 — RAG Q&A with clause citations
│   ├── gap_detector.py        # Tab 2 — Policy vs regulation gap analysis
│   ├── policy_builder.py      # Tab 3 — Policy guidance + Excel export
│   └── breach_agent.py        # Breach simulation + notification draft
│
├── core/
│   ├── __init__.py
│   └── vectorstore.py         # ChromaDB setup + HuggingFace embeddings
│
├── data/
│   └── uploads/               # Uploaded PDFs stored here
│
├── requirements.txt
├── .env                      
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12 (recommended: 3.11)
- A free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ariras.git
cd ariras
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First install downloads the HuggingFace embedding model (~80MB). This happens once automatically.

### Step 3 — Set up environment variables

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_groq_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
```

### Step 4 — Create required folders

```bash
# Windows
mkdir data\uploads
type nul > agents\__init__.py
type nul > core\__init__.py

# Mac / Linux
mkdir -p data/uploads
touch agents/__init__.py core/__init__.py
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How to Use

### Tab 1 — Regulation Q&A

1. Upload any regulation PDF (DPDP Act, SEBI circular, RBI guideline, etc.)
2. Click **Process & Index** — ARIRAS chunks and embeds the document
3. Type any compliance question
4. Get a precise answer with exact clause references

**Example questions:**
- *"What are the breach reporting obligations?"*
- *"What is the penalty for non-compliance?"*
- *"What consent requirements apply to data collection?"*

---

### Tab 2 — Policy Gap Detector

1. Index a regulation in Tab 1 first
2. Upload your company's existing policy document (PDF or TXT)
3. Click **Run Gap Analysis**
4. ARIRAS returns:
   - Compliance score (0–100%)
   - List of gaps with severity (HIGH / MEDIUM / LOW)
   - List of obligations already met

---

### Tab 3 — Policy Builder

1. Answer 3 sections of plain-English questions:
   - **Your Business** — what does your company do?
   - **Information & People** — what flows through your business?
   - **Where You Are Today** — what are your compliance concerns?
2. Click **Generate My Policy Guidance**
3. Download the Excel report with:
   - Section-by-section guidance
   - Sample clauses to adapt
   - Regulation references
   - Priority actions
   - Readiness score

> Works for **any regulation** — ARIRAS adapts based on what you uploaded and what your business does.

---

### Reporting / Dashboard (Sidebar)

- Compliance score gauge
- Gap severity breakdown (pie chart)
- Top violated obligations (bar chart)
- Compliance score over time (trend line)
- Regulation coverage table
- Export full compliance report as JSON

---

### Audit Trail (Sidebar)

- Every action logged with timestamp
- Agent runs, documents indexed, queries made
- Export as JSON for regulatory traceability

---

## Supported Regulations

ARIRAS is **regulation-agnostic** — it works with any PDF you upload. Tested with:

| Regulation | Country | Domain |
|---|---|---|
| DPDP Act 2023 | India | Data Protection |
| SEBI Circulars | India | Capital Markets |
| RBI Guidelines | India | Banking / Fintech |
| Companies Act 2013 | India | Corporate Governance |
| GDPR | EU | Data Protection |
| SOX | USA | Financial Reporting |
| AML / BSA | USA | Anti-Money Laundering |

---

## Key Features

| Feature | Description |
|---|---|
| **Regulation-agnostic RAG** | Upload any regulation PDF — ARIRAS learns it instantly |
| **Clause-level citations** | Every answer references exact clauses from the document |
| **Policy gap analysis** | Maps your policy against regulation obligations automatically |
| **Plain-English guidance** | Tells companies what to include, not just what the law says |
| **Excel report export** | Downloadable guidance with sample clauses and priority actions |
| **Full audit trail** | Every AI decision logged — exportable for compliance evidence |
| **Zero hardcoded rules** | All intelligence comes from the uploaded regulation |
| **India-first design** | Built for Indian enterprises, Indian regulations, Indian context |

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com | ✅ Yes |
| `CHROMA_PERSIST_DIR` | Path to store ChromaDB vector data | Optional (defaults to `./data/chroma_db`) |

---

## Requirements

```
streamlit>=1.32.0
langchain>=0.1.16
langchain-community>=0.0.36
langchain-chroma>=0.1.0
langchain-groq>=0.1.6
langchain-huggingface>=0.0.3
groq>=0.9.0
chromadb>=0.4.24
pypdf>=4.2.0
pdfplumber>=0.11.0
python-dotenv>=1.0.1
sentence-transformers>=3.0.0
plotly>=5.20.0
openpyxl>=3.1.2
tiktoken>=0.7.0
```

---

## Impact Model

| Metric | Estimate |
|---|---|
| Target enterprises in India | 6.3 crore MSMEs + 1,400+ listed companies |
| Cost of compliance consulting | ₹2–80 lakh per engagement (Small to Mid Sized) |
| Cost of compliance consulting | ₹ 80 + lakh per engagement (Large Sized) 
| ARIRAS cost | Entire Project costs 5000 INR per hour (Typically takes 50 hours) |
| Time to first compliance insight | Under 60 seconds |

---

## Built With

- [Streamlit](https://streamlit.io) — UI framework
- [LangChain](https://langchain.com) — Agent orchestration
- [Groq](https://groq.com) — LLM inference (Llama 3.1 70B)
- [ChromaDB](https://trychroma.com) — Vector database
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — Embeddings
- [Plotly](https://plotly.com) — Dashboard charts
- [openpyxl](https://openpyxl.readthedocs.io) — Excel generation

---

