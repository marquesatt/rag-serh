# Official Vertex AI RAG Documentation Summary

## Complete Guide to Forcing Model to Use RagCorpora Only

Based on official Google Cloud and Vertex AI documentation as of February 2026.

---

## 1. VertexRagStore Configuration - Primary Method

### Configuration Parameters

The `VertexRagStore` is the primary configuration for forcing the model to use specific corpora.

```json
{
  "tools": {
    "retrieval": {
      "disable_attribution": false,
      "vertex_rag_store": {
        "rag_resources": {
          "rag_corpus": "projects/{project}/locations/{location}/ragCorpora/{rag_corpus}"
        },
        "similarity_top_k": 20,
        "vector_distance_threshold": 0.5
      }
    }
  }
}
```

### VertexRagStore Parameters:

| Parameter | Type | Description | Purpose |
|-----------|------|-------------|---------|
| `rag_corpus` | string | RagCorpora resource name | **Specifies the exact corpus to use** |
| `rag_resources.rag_corpus` | string | Format: `projects/{project}/locations/{location}/ragCorpora/{rag_corpus}` | Restricts retrieval to single corpus only |
| `rag_resources.rag_file_ids` | list | List of specific RagFile resources | Optional: restrict to specific files only |
| `similarity_top_k` | int32 | Number of contexts to retrieve | Controls how many relevant documents are returned |
| `vector_distance_threshold` | double | Distance threshold for vector similarity | Returns only contexts with distance smaller than threshold |
| `vector_similarity_threshold` | double | Similarity score threshold | Returns only contexts with similarity larger than threshold |

---

## 2. RagTool Configuration - Force Tool Use

### Forced Function Calling Mode

To FORCE the model to ONLY use the RAG tool and not provide answers without consulting the corpus:

```python
response = model.generate_content(
    contents = [
      Content(
        role="user",
        parts=[
            Part.from_text("What is the weather like in Boston?"),
        ],
      )
    ],
    generation_config = GenerationConfig(temperature=0),
    tools = [
      Tool(
        function_declarations=[rag_retrieval_func],  # Only RAG tool
      )
    ],
    tool_config=ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            # Mode 'ANY' forces the model to ALWAYS predict function calls
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
            # Restrict to only RAG function
            allowed_function_names=["retrieve_from_corpus"],
        )
    )
)
```

### Function Calling Modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **ANY** | Model **MUST always** predict function calls (RAG in this case) | **Force RAG-only responses** |
| AUTO | Model decides between function calls or text (default) | Flexible RAG usage |
| VALIDATED | Model constrained to function calls or text, ensures schema adherence | Schema validation required |
| NONE | Model prohibited from making function calls | Disable RAG temporarily |

**CRITICAL:** Use `Mode.ANY` with `allowed_function_names=["retrieve_from_corpus"]` to force ONLY RAG usage.

---

## 3. Complete Generation Example with Corpus Forcing

### REST API - Generation with RagCorpora Only

```json
POST https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.5-flash:generateContent

{
  "contents": {
    "role": "user",
    "parts": {
      "text": "What information is available in the knowledge base about {TOPIC}?"
    }
  },
  "tools": {
    "retrieval": {
      "disable_attribution": false,
      "vertex_rag_store": {
        "rag_resources": {
          "rag_corpus": "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
        },
        "similarity_top_k": 20,
        "vector_distance_threshold": 0.5
      }
    }
  }
}
```

### Python SDK - Guaranteed Corpus-Only Responses

```python
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    ToolConfig,
    Content,
    Part,
    GenerationConfig,
    FunctionDeclaration
)

# Define RAG retrieval tool
rag_retrieval_func = FunctionDeclaration(
    name="retrieve_from_corpus",
    description="Retrieves information from the knowledge base corpus",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query for the knowledge base"
            }
        },
        "required": ["query"]
    }
)

# Create tool with RAG configuration
rag_tool = Tool(function_declarations=[rag_retrieval_func])

# Create model instance
model = GenerativeModel("gemini-2.5-flash")

# Configuration to force RAG-only responses
rag_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        allowed_function_names=["retrieve_from_corpus"]
    )
)

# Generate content - FORCED to use RAG
response = model.generate_content(
    contents=[
        Content(
            role="user",
            parts=[
                Part.from_text("What information do you have about the topic?")
            ]
        )
    ],
    tools=[rag_tool],
    tool_config=rag_config,
    generation_config=GenerationConfig(temperature=0)
)
```

---

## 4. Retrieval API - Direct Corpus Query

### Retrieve Contexts Before Generation

```json
POST https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}:retrieveContexts

{
  "vertex_rag_store": {
    "rag_resources": {
      "rag_corpus": "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
    },
    "similarity_top_k": 20,
    "vector_distance_threshold": 0.5
  },
  "query": {
    "text": "What is your question?",
    "similarity_top_k": 20
  }
}
```

**Key Point:** This API allows you to retrieve contexts from the corpus FIRST before passing them to the model, ensuring 100% corpus-based responses.

---

## 5. System Instructions for RAG-Only Behavior

### Prompt Techniques to Ensure Corpus Search First

While system instructions don't fully prevent jailbreaks, they guide the model to search the corpus first:

```python
system_instruction = """
You are a knowledge base assistant. You MUST search the provided knowledge corpus for information to answer user questions.

CRITICAL RULES:
1. ALWAYS search the knowledge corpus first for relevant information
2. ONLY provide information that is found in the knowledge corpus
3. If information is not available in the corpus, explicitly state: "This information is not available in the knowledge base"
4. NEVER attempt to answer from general knowledge when corpus information is available
5. Cite the source documents for all information provided
6. If the corpus contains relevant information, you MUST use it

For every user query:
- Search the knowledge base corpus using the retrieval tool
- Wait for the retrieved contexts
- Base your answer ONLY on the retrieved information
- Provide citations to the source documents
"""

response = model.generate_content(
    contents=user_prompt,
    system_instruction=system_instruction,
    tools=[rag_tool],
    tool_config=rag_config
)
```

---

## 6. RagCorpora Resource Specifications

### Create RagCorpus with Restrictions

```json
POST https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora

{
  "display_name": "My Knowledge Base",
  "description": "Exclusive knowledge base for RAG",
  "corpus_type_config": {
    "document_corpus": {}
  },
  "rag_vector_db_config": {
    "rag_managed_db": {
      "knn": {}
    }
  }
}
```

### Retrieval Configuration Options

```json
{
  "vertex_rag_store": {
    "rag_resources": {
      "rag_corpus": "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
    },
    "similarity_top_k": 20,
    "hybrid_search": {
      "alpha": 0.5
    },
    "filter": {
      "vector_distance_threshold": 0.5
    },
    "ranking": {
      "rank_service": {
        "model_name": "semantic-ranker-512@latest"
      }
    }
  }
}
```

---

## 7. Comparison: Optional vs Mandatory RAG

### Optional RAG (Mode.AUTO)
```python
# Model can choose to use RAG or answer from general knowledge
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
    )
)
```

### Mandatory RAG (Mode.ANY)
```python
# Model MUST use RAG for every response
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        allowed_function_names=["retrieve_from_corpus"]
    )
)
```

---

## 8. Best Practices for Ensuring Corpus-Only Responses

### 1. **Use Mode.ANY with Function Calling**
- Forces the model to ALWAYS call the retrieval function
- Prevents direct answers without corpus consultation

### 2. **Restrict to Single Corpus**
```python
"rag_resources": {
  "rag_corpus": "projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_CORPUS_ID}"
}
# Do NOT provide multiple corpus options
```

### 3. **Set Low Temperature**
```python
generation_config=GenerationConfig(temperature=0)
# Temperature 0 = more deterministic, follows tool use strictly
```

### 4. **Use Similarity Thresholds**
```python
"similarity_top_k": 20,
"vector_distance_threshold": 0.5
# Only retrieve highly relevant contexts
```

### 5. **Combine with Retrieval API**
```python
# Option 1: Retrieve contexts first
contexts = client.retrieve_contexts(
    vertex_rag_store=rag_store,
    query=user_query
)

# Option 2: Then pass to model with those contexts only
response = model.generate_content(
    contents=[...contexts...],
    system_instruction="Answer based ONLY on the provided contexts"
)
```

### 6. **System Instructions**
```python
system_instruction = """
ANSWER ONLY from the provided knowledge base.
Do NOT use general knowledge.
If information is not in the knowledge base, say so explicitly.
"""
```

---

## 9. Grounding Configuration

### Define Grounding in Studio

1. **Go to Create Prompt** → Vertex AI Studio
2. **Select Grounding: Your data**
3. **Select RAG Engine grounding source**
4. **From Corpus list, select your corpus name**
5. **Set Top-K Similarity** (default: 20)
6. **Click Save**

This ensures all model responses are grounded to your RagCorpora.

---

## 10. Complete End-to-End Example

### Corpus-Only RAG Pipeline

```python
from vertexai.generative_models import GenerativeModel, ToolConfig

# Step 1: Define the corpus constraint
corpus_id = "projects/my-project/locations/us-central1/ragCorpora/my-corpus"

rag_store_config = {
    "vertex_rag_store": {
        "rag_resources": {
            "rag_corpus": corpus_id
        },
        "similarity_top_k": 20,
        "vector_distance_threshold": 0.5
    }
}

# Step 2: Force function calling mode
tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        allowed_function_names=["retrieve_from_corpus"]
    )
)

# Step 3: System instruction
system_instruction = """
You are a knowledge base assistant bound to a specific corpus.
ALWAYS retrieve from the knowledge base.
NEVER answer from general knowledge.
Cite sources for all information.
"""

# Step 4: Create tool configuration
tool = Tool(retrieval=rag_store_config)

# Step 5: Generate response
model = GenerativeModel("gemini-2.5-flash")
response = model.generate_content(
    contents="What is your question?",
    tools=[tool],
    tool_config=tool_config,
    system_instruction=system_instruction,
    generation_config=GenerationConfig(temperature=0)
)

print(response.text)
```

---

## 11. Key Parameters Summary

### MANDATORY PARAMETERS for Corpus-Only Responses:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `tool_config.function_calling_config.mode` | `ANY` | Force tool use |
| `tool_config.function_calling_config.allowed_function_names` | `["retrieve_from_corpus"]` | Only allow RAG function |
| `rag_resources.rag_corpus` | Single corpus ID | Restrict to one corpus |
| `generation_config.temperature` | 0 | Deterministic behavior |
| `system_instruction` | Corpus-only rules | Guide model behavior |

### OPTIONAL BUT RECOMMENDED PARAMETERS:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `similarity_top_k` | 5-20 | Control relevance |
| `vector_distance_threshold` | 0.3-0.7 | Filter results |
| `hybrid_search.alpha` | 0.5 | Balance dense/sparse |
| `ranking.llm_ranker.model_name` | ranker model | Improve ranking |

---

## 12. Important Notes from Official Documentation

### From RAG Engine API Documentation:
- **VertexRagStore** is the PRIMARY configuration point for RAG
- **RagTool parameters** in `functionCallingConfig` control whether RAG is optional (AUTO) or mandatory (ANY)
- **Grounding configuration** ensures responses are tethered to corpus data
- **Tool use forcing/requirements** are implemented via `FunctionCallingConfig.Mode`

### From Function Calling Documentation:
- **Mode.ANY**: "The model is constrained to **always predict one or more function calls**"
- This is the ONLY way to guarantee the model will search the corpus
- Without Mode.ANY, the model may choose to answer from general knowledge

### From Grounding Documentation:
- Grounding "tethers output to data and reduces chances of inventing content"
- Vertex AI RAG Engine is the "configurable managed RAG service"
- When grounded, models provide "auditability by providing grounding support (links to sources)"

---

## 13. What Google Cloud Documentation Does NOT Provide

⚠️ **Important Limitation:**

The official documentation states:
> "System instructions can help guide the model to follow instructions, but they don't fully prevent jailbreaks or leaks."

**Therefore:**
- System instructions alone CANNOT force corpus-only responses
- Must combine with `Mode.ANY` function calling configuration
- Must restrict `allowed_function_names` to only RAG retrieval
- Must use low temperature setting
- Consider validating responses against corpus on backend

---

## 14. Official Resource Links

1. **RAG Engine API**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/rag-api
2. **Function Calling**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
3. **Grounding Overview**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview
4. **Ground Responses Using RAG**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/grounding/ground-responses-using-rag
5. **System Instructions**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instruction-introduction

---

## Summary: How to Guarantee Corpus-Only Responses

```
┌─────────────────────────────────────────────────────────────────┐
│ GUARANTEED CORPUS-ONLY RESPONSE CONFIGURATION                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. Use ToolConfig with Mode.ANY                                │
│    └─ Forces model to ALWAYS call retrieval function            │
│                                                                 │
│ 2. Restrict allowed_function_names                             │
│    └─ Only allows: ["retrieve_from_corpus"]                    │
│                                                                 │
│ 3. Configure VertexRagStore                                    │
│    └─ Single corpus_id only                                    │
│    └─ Set similarity_top_k and thresholds                      │
│                                                                 │
│ 4. Use System Instruction                                      │
│    └─ "Answer ONLY from provided knowledge base"               │
│    └─ "Do NOT use general knowledge"                           │
│                                                                 │
│ 5. Set temperature=0                                           │
│    └─ Ensures deterministic, rule-following behavior           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Documentation Last Updated:** February 1, 2026  
**Sources:** Official Google Cloud and Vertex AI Documentation
