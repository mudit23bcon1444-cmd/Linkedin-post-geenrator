# LinkedIn Post Generator

A few-shot LLM-powered LinkedIn post generator that crafts authentic, high-impact posts mirroring a specific author's voice, hook structure, and cadence without fine-tuning or vector databases.

> [!NOTE]
> **Sample Seed Data**: The posts included in `data/raw_posts.json` represent sample seed data for a fictional computer science student preparing for placements. They are meant to be replaced with your own historical posts.

---

## Architecture

The system operates in two decoupled stages: offline data enrichment and runtime few-shot generation.

```
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 1: ENRICHMENT (Offline)               │
│                                                             │
│   data/raw_posts.json                                       │
│          │                                                  │
│          ▼                                                  │
│   preprocess.py ────► Python: line_count calculation        │
│          │      ────► LLM Call 1: Language & Raw Tags       │
│          │      ────► LLM Call 2: Unified Title-Case Tags   │
│          ▼                                                  │
│   data/processed_posts.json                                 │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: RETRIEVAL & SYNTHESIS (Online)        │
│                                                             │
│   User Input (Topic, Length, Language)                      │
│          │                                                  │
│          ▼                                                  │
│   retriever.py (4-Tier Fallback Chain)                      │
│     Tier 1: tag + language + length                         │
│     Tier 2: tag + language                                  │
│     Tier 3: tag only                                        │
│     Tier 4: none                                            │
│          │                                                  │
│          ▼                                                  │
│   generator.py                                              │
│     - Few-shot Prompt Construction                          │
│     - Groq LLM Inference (Temperature 0.7)                  │
│     - Output Sanitization & Preamble Stripping              │
│          │                                                  │
│          ▼                                                  │
│   app.py (Streamlit UI on http://localhost:8501)            │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Few-Shot In-Context Learning**: Injects up to 2 top-engagement posts from the author as reference examples directly into the prompt.
- **Strict Line-Length Enforcement**:
  - **Short**: 1 to 5 lines
  - **Medium**: 6 to 10 lines
  - **Long**: 11 to 15 lines
- **Bilingual Support**:
  - **English**: Clean, punchy corporate & engineering English.
  - **Hinglish**: Natural colloquial blend of Hindi and English words, strictly constrained to Roman Latin script.
- **Engagement-Prioritized Fallback**: If an exact match (`tag + language + length`) is not available, the system gracefully falls back to `tag + language`, then `tag only`, sorting candidates by engagement count.

---

## Project Structure

```
linkedin-post-generator/
├── app.py                      # Streamlit web user interface
├── config.py                   # Central constants, lengths, and mappings
├── llm.py                      # Groq LLM client and model smoke test
├── preprocess.py               # Stage 1: metadata enrichment & tag unification
├── retriever.py                # Stage 2a: Pandas-backed 4-tier example retriever
├── generator.py                # Stage 2b: Prompt engineering and post generator
├── data/
│   ├── raw_posts.json          # 24 seed author posts
│   └── processed_posts.json    # Preprocessed dataset with tags & line counts
├── tests/
│   ├── test_config.py          # Boundary and mapping unit tests
│   ├── test_retriever.py       # Fallback hierarchy & sorting tests
│   ├── test_generator.py       # Prompt & preamble stripping tests
│   └── test_preprocess.py      # Extraction, parsing, & retry unit tests
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies (unpinned)
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
└── README.md                   # Documentation
```

---

## Setup & Installation

### 1. Prerequisites
- Python 3.11 or 3.12 installed.
- A free Groq API key from [Groq Console](https://console.groq.com).

### 2. Virtual Environment & Dependencies
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Populate `.env` with your Groq credentials:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

---

## How to Run

### Step 1: Preprocess Data (Stage 1)
Enrich raw posts with line counts, language labels, and unified topic tags:
```bash
python preprocess.py
```

### Step 2: Run the Web App (Stage 2)
Start the Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## How to Swap in Your Own Posts

1. Open `data/raw_posts.json`.
2. Replace the entries with your historical LinkedIn posts in the format:
   ```json
   [
     {
       "text": "Your post hook line.\nBody text.\nTakeaway question?",
       "engagement": 450
     }
   ]
   ```
3. Re-run `python preprocess.py` to re-index and re-cluster tags.
4. Refresh your Streamlit app!

---

## Running Unit Tests

The test suite runs entirely offline (mocking the LLM):
```bash
pytest -q
```
All 19 tests validate config boundary conditions, retriever fallback layers, prompt construction, preamble stripping, retry loops, and JSON fence extraction.

---

## Known Limitations

- **Exact-Match Retrieval**: Retrieval uses exact categorical equality on tags rather than semantic vector similarity.
- **Small Dataset**: The default seed dataset contains 24 posts, meaning some granular combinations (e.g. Hinglish Long Mental Health) trigger fallback tiers.
- **LLM-Inferred Tags**: Topic tags and language classifications are inferred by the LLM during Stage 1 and may vary depending on model temperature.

---

## Future Work

- **Database Storage**: Migrate from static JSON files to PostgreSQL / SQLite for incremental post additions and author management.
- **Multi-Author Profiles**: Support multiple personas/creators selectable via dropdown.
- **Engagement-Weighted Selection**: Implement weighted probabilistic sampling so top posts are favored while maintaining variety.
- **Direct LinkedIn API Publishing**: Add OAuth integration to publish or schedule drafts directly to LinkedIn.
