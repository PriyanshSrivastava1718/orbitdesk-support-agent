# OrbitDesk Support Agent

A local AI-powered customer-support agent built for the **Tantrabodh AI Engineer Internship Assignment**.

OrbitDesk Support Agent combines **Retrieval-Augmented Generation (RAG)**, **four-way triage**, **LangGraph orchestration**, **response verification**, bounded retry/revision, safe failure, and **JSON-schema-validated structured output**.

The application runs its language models locally and does not rely on hosted LLM APIs for runtime response generation.

---

## Overview

A customer-support assistant should not blindly send every request through a generation pipeline.

OrbitDesk first triages an incoming request into one of four routes:

- `answerable`
- `requires_clarification`
- `requires_escalation`
- `out_of_scope`

Only answerable requests enter the RAG pipeline.

For an answerable request, relevant evidence is retrieved from the supplied OrbitDesk knowledge base and historical resolved cases. A local generation model produces an evidence-grounded response, which is then verified before being returned.

If verification fails, the graph allows one revision attempt. If the revised response still fails verification, the system returns a safe failure rather than presenting an unsupported answer.

---

## Architecture

```text
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
      RAG              Clarification            Human
       |                  Response              Support
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

Out-of-scope requests are routed directly to an out-of-scope response.
```

The workflow is orchestrated using **LangGraph** with shared typed state, deterministic routing functions, conditional edges, and bounded retry behaviour.

---

## RAG Pipeline

For answerable requests, OrbitDesk uses Retrieval-Augmented Generation.

```text
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

The query and document chunks are embedded using the same Sentence Transformer model.

Cosine similarity ranks the stored chunks according to their semantic similarity with the query. The highest-ranked chunks are supplied to the local generation model as evidence.

Current knowledge-base documentation is treated as authoritative. Historical resolved cases are used as supporting evidence.

---

## Triage

Triage occurs **before RAG**.

### `answerable`

The request contains enough information to follow a documented OrbitDesk support path.

The request proceeds into the RAG pipeline.

### `requires_clarification`

The request concerns OrbitDesk, but information required to safely determine the appropriate support path is missing.

The system requests additional information instead of blindly generating an answer.

### `requires_escalation`

The available context indicates that the issue requires human support or has reached an escalation condition.

### `out_of_scope`

The request falls outside the supported OrbitDesk customer-support domain.

LangGraph conditional edges route each classification to the corresponding node.

---

## Verification

Generated answers are not automatically trusted.

For the answerable route, the generated response is checked against the retrieved OrbitDesk evidence.

Verification includes deterministic safeguards and local model-based grounding checks.

Deterministic checks ensure that:

- retrieved evidence exists
- source identifiers are available
- the generated answer is not empty

The grounding check evaluates whether the generated response:

- is supported by retrieved evidence
- contradicts the available evidence
- invents OrbitDesk product behaviour
- invents unsupported troubleshooting instructions

A response that passes verification proceeds to the final output.

A response that fails verification is routed to the revision node.

---

## Retry and Safe Failure

The graph implements a bounded retry path:

```text
Generation
    |
    v
Verification
    |
    +---- PASS ----> Final Response
    |
   FAIL
    |
    v
 Revision
    |
    v
Verification
    |
    +---- PASS ----> Final Response
    |
   FAIL
    |
    v
Safe Failure
```

Only **one revision attempt** is permitted.

The workflow maintains a `retry_count`, and the LangGraph invocation also specifies a recursion limit as an additional safeguard against infinite graph execution.

The following paths have been manually exercised during integration testing:

```text
triage -> rag -> verifier -> END

triage -> rag -> verifier -> revision -> verifier -> END

triage -> rag -> verifier -> revision -> verifier
       -> safe_failure -> END
```

---

## Structured Output

The completed internal graph state is converted into the response contract supplied in:

```text
data/output_schema.json
```

The required fields are:

- `classification`
- `answer`
- `sources`
- `confidence`
- `requires_human`
- `reason`

Example:

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

The final response is programmatically validated against the supplied JSON Schema using `jsonschema`.

Schema validation has been successfully exercised on a real graph output.

`confidence` is an application-level indicator and should not be interpreted as a calibrated probability.

---

## Models

### Embedding Model

**sentence-transformers/all-MiniLM-L6-v2**

Used for:

- document embeddings
- query embeddings
- semantic retrieval

### Local Generation Model

**Qwen/Qwen2.5-1.5B-Instruct**

Used for:

- support-response generation
- triage classification
- semantic grounding verification

The application uses locally executed models rather than remotely hosted OpenAI, Anthropic, Gemini, or other LLM APIs.

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- LangGraph
- JSON Schema
- Pytest
- Cosine Similarity
- Qwen2.5 Instruct

---

## Project Structure

```text
orbitdesk-support-agent/
|
+-- data/
|   +-- output_schema.json
|   +-- resolved_cases.json
|   +-- sample_questions.json
|
+-- knowledge_base/
|   +-- 01_product_overview.md
|   +-- 02_roles_and_permissions.md
|   +-- 03_workspace_settings_and_timezones.md
|   +-- 04_scheduled_exports.md
|   +-- 05_api_credentials.md
|   +-- 06_connections_and_refreshes.md
|   +-- 07_delivery_destinations.md
|   +-- 08_escalation_and_diagnostics.md
|   +-- 09_audit_logs.md
|   +-- 10_security_and_safe_responses.md
|
+-- src/
|   +-- chunk_data.py
|   +-- graph.py
|   +-- load_data.py
|   +-- output_formatter.py
|   +-- rag.py
|   +-- retrieval.py
|   +-- test_embeddings.py
|   +-- test_llm.py
|   +-- triage.py
|   +-- verifier.py
|
+-- tests/
|   +-- test_graph_routing.py
|
+-- .gitignore
+-- LICENSE
+-- README.md
+-- requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd orbitdesk-support-agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The required Hugging Face models are downloaded automatically on first use. Initial execution can therefore take longer depending on network speed and hardware.

---

## Running the Agent

From the repository root:

```bash
python src/graph.py
```

Node execution is logged to the terminal, for example:

```text
NODE: triage
NODE: rag
NODE: verifier
```

The application then displays the result and the final structured JSON output.

---

## Testing

Run the automated routing test using:

```bash
pytest tests/test_graph_routing.py -v
```

The automated test verifies routing logic without depending on the exact wording produced by the language model.

Integration testing has also exercised:

1. directly answerable requests
2. ambiguous requests requiring clarification
3. requests requiring escalation
4. out-of-scope requests
5. successful answer verification
6. initial verification failure followed by revision
7. repeated verification failure followed by safe failure
8. bounded retry behaviour
9. structured-output JSON-schema validation

The assignment's multi-document evidence case should also be included in the final recorded/sample test evidence.

---

## Example Execution

Input:

```text
Can a Viewer create an API credential?
```

Triage:

```text
answerable
```

Observed graph execution:

```text
NODE: triage
NODE: rag
NODE: verifier
```

Generated response:

```text
No, a Viewer cannot create an API credential in OrbitDesk.
Viewers require an Admin or Owner role to access developer
settings and create API credentials.
```

Retrieved sources included:

```text
CASE-1058
KB-005
KB-005
```

Verification:

```text
Verification passed: True
Verification reason: Answer is grounded in the retrieved evidence.
Retry count: 0
```

The resulting structured response also passed validation against `data/output_schema.json`.

---

## Hardware Used

The project was developed and tested locally on a Windows laptop with:

- **System RAM:** 16 GB
- **Dedicated GPU VRAM:** 4 GB
- **GPU:** NVIDIA discrete GPU
- **Operating System:** Windows

During observed execution, approximately **3.2 GB of dedicated GPU memory** was allocated while the local models were loaded.

Depending on available VRAM, Transformers/Accelerate may place or offload model parameters across GPU, CPU, and system memory.

---

## Current Status

Core implementation:

- [x] Source-document loading
- [x] Chunking
- [x] Embedding generation
- [x] Query embedding
- [x] Cosine-similarity retrieval
- [x] RAG generation
- [x] Four-way triage
- [x] LangGraph shared state
- [x] Conditional routing
- [x] Response verification
- [x] Retry/revision path
- [x] Re-verification
- [x] Safe-failure path
- [x] Infinite-loop protection
- [x] Automated routing test
- [x] Structured output
- [x] JSON-schema validation

Submission assets:

- [ ] Final required test/sample outputs
- [ ] Graph diagram (PNG/JPG)
- [ ] Final repository compliance review
- [ ] 4–7 minute walkthrough video

---

## Known Limitations

The implementation is intentionally lightweight and scoped to the internship assignment.

Current limitations include:

- Generation and semantic verification use the same local Qwen model, so verification is not fully independent from generation.
- Multiple components currently initialize model instances independently, increasing memory usage and startup overhead.
- Verification provides a grounding safeguard rather than a formal factual guarantee.
- Retrieval quality depends on the supplied knowledge base and embedding model.
- Confidence values are application-level indicators rather than statistically calibrated probabilities.
- Local inference may require CPU offloading on GPUs with limited VRAM.

These are potential future optimizations rather than additional features required by the current prototype.

---

## AI Assistance Disclosure

AI coding assistants, including **ChatGPT and Claude**, were used during development for:

- concept explanation
- architecture discussion
- debugging assistance
- code review
- testing guidance
- documentation assistance

The implementation was executed, tested, integrated, and reviewed locally.

AI coding assistants are development tools only. The OrbitDesk application itself does **not** depend on ChatGPT, Claude, Gemini, or another hosted LLM API for runtime response generation.

---

## Security

No API keys, credentials, authentication secrets, or customer secrets are included in the repository.

The local virtual environment, Python caches, and temporary development files are excluded through `.gitignore`.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

**Priyansh Srivastava**

B.Tech Computer Science and Engineering  
Pranveer Singh Institute of Technology, Kanpur
