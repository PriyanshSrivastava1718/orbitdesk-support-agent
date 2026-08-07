# OrbitDesk Support Agent

A local, AI-powered customer-support agent built for the **Tantrabodh AI Engineer Internship Assignment**.

OrbitDesk Support Agent combines **Retrieval-Augmented Generation (RAG)**, **four-way triage**, **LangGraph orchestration**, **response verification**, **bounded retry/revision**, **safe failure**, and **JSON-schema-validated structured output**.

The application runs its language models locally. No hosted LLM API (OpenAI, Anthropic, Gemini, or otherwise) is used for runtime response generation.

---

## Overview

A support assistant should not blindly send every request through a generation pipeline. OrbitDesk first **triages** an incoming request into one of four routes:

- `answerable`
- `requires_clarification`
- `requires_escalation`
- `out_of_scope`

Only `answerable` requests enter the RAG pipeline. For those, relevant evidence is retrieved from the supplied OrbitDesk knowledge base and historical resolved cases, a local model generates an evidence-grounded response, and that response is **verified** before being returned.

If verification fails, the graph allows **one revision attempt**. If the revised response still fails verification, the system returns a **safe failure** rather than presenting an unsupported answer.

---

## Architecture

```
                         User Query
                             |
                             v
                           Triage
                             |
       +---------------------+---------------------+
       |                     |                     |
       v                     v                     v
   Answerable        Requires Clarification   Requires Escalation
       |                     |                     |
       v                     v                     v
      RAG              Clarification            Escalation
       |                  Response               Response
       v                     |                     |
   Retrieval                 v                     v
       |                    END                   END
       v
 Augmentation
       |
       v
  Generation
       |
       v
   Verifier
       |
   +---+---+
   |       |
  PASS    FAIL
   |       |
   v       v
  END   Revision
           |
           v
        Verifier
           |
       +---+---+
       |       |
      PASS    FAIL
       |       |
       v       v
      END   Safe Failure
                |
                v
               END

Out-of-scope requests are routed directly to an out-of-scope response -> END.
```

The workflow is orchestrated with **LangGraph**: a shared typed state (`AgentState`), deterministic routing functions (`route_after_triage`, `route_after_verification`), conditional edges, and bounded retry behaviour.

A rendered graph diagram is included at `docs/graph.png` (see [Graph Diagram](#graph-diagram)).

---

## RAG Pipeline

```
Source Documents
      |
      v
   Chunking
      |
      v
MiniLM Embeddings
      |
      v
Document Vectors

User Query
      |
      v
Query Embedding
      |
      v
Cosine Similarity
      |
      v
Top-K Retrieval
      |
      v
Retrieved Evidence + Query
      |
      v
Local Qwen Generation
      |
      v
Generated Answer
```

The query and document chunks are embedded with the same Sentence Transformer model. Cosine similarity ranks stored chunks by semantic similarity to the query; the top-K chunks (default `k = 3`) are supplied to the local generation model as evidence.

Current knowledge-base documentation is treated as **authoritative**. Historical resolved cases are used as **supporting evidence only** — a resolved case can outrank a KB document on similarity score without being more authoritative. Cases marked `superseded` (e.g. `CASE-0914`, which describes a legacy personal-API-token flow removed in OrbitDesk 4.0) are excluded deterministically at retrieval time, regardless of similarity score, so an outdated case can never be surfaced as current guidance.

---

## Triage

Triage runs **before** RAG and classifies every request into exactly one of four routes.

| Label | Meaning |
|---|---|
| `answerable` | The request contains enough information to follow a documented OrbitDesk support path. Proceeds to RAG. |
| `requires_clarification` | Related to OrbitDesk, but missing information needed to safely determine the right path. |
| `requires_escalation` | Documented troubleshooting has already been attempted and failed, or the situation matches a documented escalation condition. |
| `out_of_scope` | Unrelated to OrbitDesk support, or asks the assistant to do something outside its scope (e.g. refunds, legal advice). |

Classification is produced by a local Qwen model given a few-shot instruction prompt, then validated deterministically in code against the four allowed labels — an invalid or malformed model output is not trusted and falls back to `requires_clarification` rather than being passed through unchecked.

LangGraph conditional edges route each classification to its corresponding node.

---

## Verification

Generated answers are not automatically trusted. For the `answerable` route, the response is checked against the retrieved evidence before being returned.

**Deterministic checks:**
- Retrieved evidence exists
- Source identifiers are available
- The generated answer is non-empty

**Model-based grounding check:** a local Qwen call evaluates whether the response is supported by the retrieved evidence, avoids inventing OrbitDesk behaviour or troubleshooting instructions, and does not contradict the supplied evidence.

A response that passes proceeds to the final output. A response that fails is routed to revision.

---

## Retry and Safe Failure

```
Generation -> Verification -> PASS -> Final Response
                  |
                 FAIL
                  |
                  v
              Revision -> Verification -> PASS -> Final Response
                               |
                              FAIL
                               |
                               v
                         Safe Failure -> Final Response
```

Only **one revision attempt** is permitted. The shared state tracks `retry_count`, and the LangGraph invocation also sets `config={"recursion_limit": 10}` as an additional safeguard against unbounded graph execution.

The revision node re-generates the answer using the verifier's failure reason as additional guidance, rather than repeating the identical prompt — this gives the revision attempt a genuine chance to produce a different, better-grounded answer instead of deterministically reproducing the same failure.

Routes exercised during integration testing:

```
triage -> rag -> verifier -> END
triage -> rag -> verifier -> revision -> verifier -> END
triage -> rag -> verifier -> revision -> verifier -> safe_failure -> END
```

---

## Structured Output

Internal graph state is converted into the response contract defined in `data/output_schema.json` via `src/output_formatter.py`, and validated programmatically with `jsonschema` before being returned.

Required fields: `classification`, `answer`, `sources`, `confidence`, `requires_human`, `reason`.

```json
{
  "classification": "answerable",
  "answer": "No, a Viewer cannot create an API credential in OrbitDesk. Viewers require an Admin or Owner role to access developer settings and create API credentials.",
  "sources": [
    {
      "source_id": "KB-005",
      "passage": "Only Owners and Admins can create or revoke credentials. Analysts and Viewers cannot create API credentials."
    }
  ],
  "confidence": 0.9,
  "requires_human": false,
  "reason": "The request was answerable using retrieved OrbitDesk evidence and the generated response passed verification."
}
```

`confidence` is an application-level indicator derived from classification and verification outcome — it is **not** a statistically calibrated probability.

---

## Models

| Role | Model | Purpose |
|---|---|---|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Document and query embeddings, semantic retrieval |
| Generation | `Qwen/Qwen2.5-1.5B-Instruct` | Support-answer generation, triage classification, semantic grounding verification |

Qwen is a single model family reused across three responsibilities (generation, triage, verification) — it is not three separate models. Exact revisions and observed load time / response latency are recorded in [Model Details](#model-details) below.

The application uses locally executed models only — no runtime dependency on a hosted LLM API.

---

## Model Details

| | Embedding model | Generation model |
|---|---|---|
| Model ID | `sentence-transformers/all-MiniLM-L6-v2` | `Qwen/Qwen2.5-1.5B-Instruct` |
| Revision (commit SHA) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |

> Revisions were obtained directly from the Hugging Face Hub (`huggingface_hub.model_info(repo_id).sha`) for the exact model versions downloaded, since no explicit `revision=` was pinned in code.

### Observed Load Time and Latency

Qwen is loaded independently by three components (`triage.py`, `rag.py`, `verifier.py`); MiniLM is loaded once by `retrieval.py`. All values below are real measurements from a full graph run on the development hardware (see [Hardware Used](#hardware-used)).

| Component | Load time | Task | Latency |
|---|---|---|---|
| MiniLM (`retrieval.py`) | 7.69s | Embed 53 KB/case chunks (one-time startup) | 1.12s |
| MiniLM (`retrieval.py`) | — | Embed one query | 0.03s |
| Qwen (`triage.py`) | 12.26s | Classification | 1.15s |
| Qwen (`rag.py`) | 5.81s | Answer generation | 57.05s |
| Qwen (`verifier.py`) | 4.01s | Grounding verification | 43.18s |

> Generation and verification latency are high relative to model size (1.5B parameters) because the development GPU has 4 GB VRAM, which is insufficient to hold the full model — some parameters are offloaded to CPU/disk during inference (`"Some parameters are on the meta device because they were offloaded to the cpu and disk."`). This is a direct, observed hardware constraint, not a code inefficiency.

---

## Technology Stack

Python, PyTorch, Hugging Face Transformers, Sentence Transformers, LangGraph, JSON Schema, Pytest, Qwen2.5 Instruct.

---

## Project Structure

```
orbitdesk-support-agent/
|
+-- data/
|   +-- output_schema.json
|   +-- resolved_cases.json
|   +-- sample_questions.json
|
+-- knowledge_base/
|   +-- 01_product_overview.md ... 10_security_and_safe_responses.md
|
+-- src/
|   +-- chunk_data.py        # KB + resolved-case chunking, metadata
|   +-- load_data.py         # initial data-loading exploration script
|   +-- retrieval.py         # embeddings + cosine-similarity top-k retrieval
|   +-- triage.py            # four-way classification
|   +-- rag.py                # retrieval + generation
|   +-- verifier.py          # deterministic + model-based verification
|   +-- output_formatter.py  # internal state -> schema-compliant output
|   +-- graph.py             # LangGraph orchestration
|
+-- tests/
|   +-- test_graph_routing.py
|
+-- docs/
|   +-- graph.png            # graph diagram
|
+-- .gitignore
+-- LICENSE
+-- README.md
+-- requirements.txt
```

---

## Setup

### 1. Clone the repository
```
git clone <repository-url>
cd orbitdesk-support-agent
```

### 2. Create a virtual environment

**Windows:**
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell blocks script execution for the current session:
```
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```
Hugging Face models are downloaded automatically on first use — initial execution may take longer depending on network speed and hardware.

---

## Running the Agent

From the repository root:
```
python src/graph.py
```

Node execution is logged to the terminal:
```
NODE: triage
NODE: rag
NODE: verifier
```

The script then prints the final state and the schema-validated structured JSON output.

To change the input question, edit `initial_state["question"]` at the bottom of `src/graph.py`.

---

## Testing

Run the automated routing test:
```
pytest tests/test_graph_routing.py -v
```
This verifies routing logic (which node/path executes for a given classification and state) without depending on exact model-generated wording.

Integration testing has exercised, end to end:

1. A directly answerable question
2. A question requiring evidence from two documents
3. An ambiguous question requiring clarification
4. An out-of-scope request
5. An initial answer that fails verification, triggers revision, and is either corrected or routed to safe failure
6. Bounded retry behaviour (never more than one revision)
7. Structured-output JSON-schema validation

Sample outputs for all required cases are in `sample_outputs/` (see [Sample Outputs](#sample-outputs)).

---

## Sample Outputs

_Link or embed the saved terminal output / JSON for each of the five required test cases here, e.g. `sample_outputs/q1_answerable.txt`, `sample_outputs/q2_multi_doc.txt`, etc._

---

## Graph Diagram

![OrbitDesk graph](docs/graph.png)

---

## Hardware Used

Developed and tested locally on a Windows laptop:

- **CPU / System RAM:** 16 GB DDR5
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- **Storage:** 512 GB SSD (~120 GB free)
- **OS:** Windows

During a full graph execution, observed resource usage was approximately:
- System RAM: ~11.3 / 16 GB
- Dedicated GPU memory: ~3.2 / 4.0 GB
- GPU temperature: ~55°C

Depending on available VRAM, Transformers/Accelerate may place or offload model parameters across GPU, CPU, and system memory (observed during this run — see [Model Details](#model-details)).

---

## Current Status

**Core implementation:**
- [x] Source-document loading
- [x] Chunking (Markdown section-aware, plus resolved-case chunking)
- [x] MiniLM embeddings, query embeddings, cosine-similarity retrieval
- [x] Superseded-case filtering (deterministic)
- [x] Local Qwen RAG generation
- [x] Four-way triage with deterministic label validation
- [x] LangGraph shared state, nodes, conditional routing
- [x] Verification (deterministic + model-based grounding)
- [x] Retry/revision path with failure-reason feedback
- [x] Bounded retry (retry_count) + recursion-limit safeguard
- [x] Safe-failure path
- [x] Structured output + JSON-schema validation
- [x] Automated routing test
- [x] Model revisions, load time, and latency recorded

**Submission assets remaining:**
- [ ] All five required test cases captured as sample outputs, including the multi-document case
- [ ] Graph diagram exported as PNG/JPG
- [ ] Exact CPU/GPU model names and defensible minimum hardware requirements
- [ ] Final repository compliance pass
- [ ] 4–7 minute walkthrough video

---

## Known Limitations

This implementation is intentionally scoped to a 3–4 hour assignment. Known limitations, honestly stated rather than hidden:

- **Generation and verification share a model.** Both use the same local Qwen instance, so verification is not fully independent of the generator. A more robust design would use an independent grounding check (e.g. embedding-similarity overlap between answer and evidence) rather than asking the same model to grade its own output.
- **Independent model loading.** `triage.py`, `rag.py`, and `verifier.py` each currently load their own Qwen instance rather than sharing one. This works on the development hardware but increases startup time and memory usage; sharing a single loaded model across nodes is a natural follow-up.
- **Triage is a small (1.5B) instruct model doing few-shot classification.** It performs reliably on the assignment's sample questions but, like any small LLM classifier, is not guaranteed to generalize perfectly to unseen phrasings.
- **Confidence values are application-level indicators**, not statistically calibrated probabilities.
- **Retrieval quality is bounded by the supplied knowledge base and a general-purpose embedding model** (not fine-tuned on OrbitDesk-specific language).
- **CPU/disk offloading occurs during inference** because the development GPU's 4 GB VRAM cannot hold the full model, which directly explains the elevated generation/verification latency (see [Model Details](#model-details)).

**With more time, I would:** decouple verification from the generation model, share a single model instance across nodes, and calibrate `confidence` against retrieval similarity and verification outcome rather than fixed per-route values.

---

## AI Assistance Disclosure

AI coding assistants — **ChatGPT and Claude** — were used throughout development for concept explanation, architecture discussion, debugging assistance, code review, testing guidance, and documentation assistance.

All code was executed, tested, integrated, and reviewed locally by the author. The OrbitDesk application itself does not depend on ChatGPT, Claude, Gemini, or any hosted LLM API for runtime response generation.

---

## Security

No API keys, credentials, authentication secrets, or customer data are included in this repository. The virtual environment, Python caches, and local temp files are excluded via `.gitignore`.

---

## License

MIT License — see `LICENSE`.

---

## Author

**Priyansh Srivastava**
B.Tech Computer Science and Engineering
Pranveer Singh Institute of Technology, Kanpur
