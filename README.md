# OrbitDesk Support Agent

A local AI-powered customer-support agent built for the **Tantrabodh AI Engineer Internship Assignment**.

OrbitDesk Support Agent uses **Retrieval-Augmented Generation (RAG)**, **triage-based routing**, **LangGraph orchestration**, and **response verification** to answer support questions using supplied OrbitDesk documentation and historical resolved cases.

The application runs its language model locally and does not rely on hosted LLM APIs for response generation.

---

## Overview

A customer-support assistant should not blindly send every request to a language model.

OrbitDesk first classifies an incoming request and determines how it should be handled.

Requests are routed into four categories:

- `answerable`
- `requires_clarification`
- `requires_escalation`
- `out_of_scope`

Answerable requests enter the RAG pipeline, where relevant evidence is retrieved from the supplied support documents before a response is generated.

The generated response is then verified against the retrieved evidence before being returned.

---

## Architecture

```text
                         User Query
                             |
                             v
                           Triage
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
         Answerable     Clarification   Escalation
              |              |              |
              v              v              v
             RAG            END            END
              |
              v
          Retrieval
              |
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
             +----+----+
             |         |
            PASS      FAIL
             |         |
             v         v
            END    Safe Failure
                       |
                       v
                      END

Out-of-scope requests are routed directly to an appropriate boundary response.
```

The workflow is orchestrated using **LangGraph**, with shared state carrying information between nodes.

---

## RAG Pipeline

For answerable requests, OrbitDesk uses a Retrieval-Augmented Generation pipeline.

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
Qwen Generation
      |
      v
Generated Answer
```

The query and document chunks are embedded using the same embedding model.

Cosine similarity is then used to rank chunks according to their semantic similarity with the query. The highest-ranked chunks are supplied to the local generation model as supporting evidence.

---

## Triage

Before entering the RAG pipeline, each request passes through a triage stage.

### `answerable`

The request contains enough information to follow a documented OrbitDesk support path.

The request proceeds to RAG.

### `requires_clarification`

The request concerns OrbitDesk, but important information required to safely answer or diagnose it is missing.

The workflow requests additional information instead of generating an unsupported answer.

### `requires_escalation`

The issue requires human support according to the available support rules or previous troubleshooting context.

### `out_of_scope`

The request falls outside the supported OrbitDesk customer-support domain.

These routes are implemented using LangGraph conditional edges.

---

## Verification

Generated answers are not automatically trusted.

After generation, the verifier checks the response against the retrieved evidence.

Verification currently combines deterministic checks with local model reasoning.

Deterministic checks include:

- retrieved evidence exists
- source identifiers are available
- the generated answer is not empty

The model-based grounding check verifies that the response:

- is supported by retrieved evidence
- does not contradict the evidence
- does not invent OrbitDesk product behaviour
- does not invent unsupported troubleshooting instructions

If verification succeeds, the answer can proceed to the final response.

If verification fails, the graph routes the response through a revision path.

---

## Retry and Safe Failure

The workflow includes bounded retry behaviour.

```text
Generation
    |
    v
Verification
    |
    +---- PASS ----> Final
    |
   FAIL
    |
    v
 Revision
    |
    v
Verification
    |
    +---- PASS ----> Final
    |
   FAIL
    |
    v
Safe Failure
```

A retry counter prevents indefinite revision loops.

The LangGraph invocation also uses a recursion limit as an additional orchestration safeguard.

If the system still cannot produce a verifiable answer after the permitted revision, it returns a safe failure instead of presenting an unsupported response as reliable.

---

## Models

### Embedding Model

**MiniLM** via Sentence Transformers / Hugging Face.

Used for:

- document embeddings
- query embeddings
- semantic retrieval

### Local Generation Model

**Qwen/Qwen2.5-1.5B-Instruct**

Used for:

- support response generation
- triage classification
- semantic grounding verification

The application uses a local generation model rather than a remotely hosted LLM API.

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- LangGraph
- Cosine Similarity
- Local Qwen Instruct model

---

## Project Structure

```text
orbitdesk-support-agent/
|
+-- src/
|   +-- retrieval.py
|   +-- rag.py
|   +-- triage.py
|   +-- verifier.py
|   +-- graph.py
|
+-- README.md
+-- .gitignore
```

The repository structure may be expanded as final tests and documentation are added.

---

## Example

Input:

```text
Can a Viewer create an API credential?
```

Triage:

```text
answerable
```

Graph execution:

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

Retrieved evidence included:

```text
CASE-1058
KB-005
```

Verification:

```text
Verification passed: True
Retry count: 0
```

---

## Current Status

The core workflow is implemented and running:

- [x] Document loading and chunking
- [x] Embedding generation
- [x] Semantic retrieval
- [x] RAG generation
- [x] Four-way triage
- [x] LangGraph orchestration
- [x] Conditional routing
- [x] Response verification
- [x] Retry/revision routing
- [x] Safe-failure routing
- [x] Infinite-loop protection
- [ ] Complete required edge-case test suite
- [ ] Automated graph-routing test
- [ ] Final structured-output validation
- [ ] Final graph diagram
- [ ] Demo walkthrough

---

## Known Limitations

The current implementation is intentionally lightweight and designed for the scope of the internship assignment.

Some current limitations include:

- Generation and semantic verification use the same local Qwen model.
- Multiple components currently initialize model instances independently, increasing memory usage and startup overhead.
- Verification provides a lightweight grounding safeguard rather than a formal factual guarantee.
- Retrieval quality depends on the supplied knowledge base and embedding model.

These are potential areas for future optimization rather than requirements for the current prototype.

---

## Setup

Detailed installation and execution instructions will be finalized after the remaining integration tests.

The project requires Python and sufficient memory to run the local embedding and generation models.

---

## Testing

The final test suite will cover:

1. A directly answerable request
2. A request requiring evidence from multiple documents
3. An ambiguous request requiring clarification
4. An out-of-scope request
5. A generated answer that initially fails verification
6. Graph-routing behaviour independent of exact generated wording

Test results and sample outputs will be added after final validation.

---

## AI Assistance Disclosure

AI coding assistants were used during development for explanation, debugging assistance, code review, and documentation support.

The architecture, implementation decisions, testing, integration, and final validation were reviewed during development rather than treating AI-generated output as automatically correct.

The application itself does **not** depend on ChatGPT, Claude, or another hosted LLM API for runtime response generation.

---

## Author

**Priyansh Srivastava**

B.Tech Computer Science and Engineering  
Pranveer Singh Institute of Technology, Kanpur
