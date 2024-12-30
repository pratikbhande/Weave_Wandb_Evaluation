# LLM Evaluation and Comparison with Weave and WandB

This repository demonstrates how to use **Weave** (an extension of Weights and Biases) to evaluate and compare large language models (LLMs). The project incorporates model evaluation pipelines, scoring mechanisms, and a dashboard to visualize the results effectively.

## Overview

This codebase provides an example of how to evaluate LLMs using contextual relevance for question answering. It uses:

- **Weave**: For defining operations ("ops") and tracking the evaluation pipeline.
- **Weights and Biases (WandB)**: For managing runs and visualizing performance metrics.
- **OpenAI API**: For leveraging LLMs to generate responses, embeddings, and contextual queries.

The goal is to compare different models’ ability to provide relevant and accurate answers based on a given context.

---

## Features

1. **Contextual Relevance**: Evaluates LLMs by ensuring their answers are derived from the provided context.
2. **Document Embedding**: Converts documents into embeddings for efficient similarity search.
3. **Retrieval-Augmented Generation (RAG)**: Implements RAG-based query answering to find the most relevant context and answer the query.
4. **Evaluation Pipeline**: Uses Weave to create custom scorers that assess the precision of context-based answers.
5. **Dashboard Integration**: Generates a dashboard to visualize and compare model performance metrics.

---

## Dependencies

- Python 3.8+
- [Weave](https://wandb.ai/weave) (Weights and Biases extension)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- NumPy
- Nest AsyncIO
- JSON

Install dependencies using:
```bash
pip install weave wandb openai numpy nest_asyncio
```

---

## Code Explanation

### 1. **Document Embedding**

The `docs_to_embeddings` function converts a list of documents into embeddings using OpenAI’s embedding models.

```python
def docs_to_embeddings(docs: list) -> list:
    openai = OpenAI()
    document_embeddings = []
    for doc in docs:
        response = (
            openai.embeddings.create(
                input=doc,
                model="text-embedding-3-small"
            )
            .data[0]
            .embedding
        )
        document_embeddings.append(response)
    return document_embeddings
```

### 2. **RAG Querying**

The `get_most_relevant_document` operation identifies the most relevant document based on a query using cosine similarity.

```python
@weave.op()
def get_most_relevant_document(query):
    # Fetch query embedding
    # Calculate cosine similarity
    # Return the most relevant document
```

### 3. **RAG Model**

The `RAGModel` class retrieves relevant documents and generates responses based on the retrieved context.

```python
class RAGModel(weave.Model):
    def predict(self, question: str) -> dict:
        context = get_most_relevant_document(question)
        # Formulate query
        # Generate response using OpenAI
        return {"answer": answer, "context": context}
```

### 4. **Evaluation with Weave**

The `context_precision_score` operation evaluates the relevance of the context in generating the answer.

```python
@weave.op()
async def context_precision_score(question, model_output):
    # Construct prompt
    # Evaluate if context was useful
    return {"verdict": int(response["verification"]) == 1}
```

### 5. **Evaluation Pipeline**

Weave manages the evaluation pipeline and integrates it with WandB’s dashboard for visualization.

```python
evaluation = weave.Evaluation(
    dataset=questions,
    scorers=[context_precision_score]
)

await evaluation.evaluate(model)
```

---

## Using the Code

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Set up your OpenAI API key in the environment:
   ```bash
   export OPENAI_API_KEY=<your-api-key>
   ```

3. Run the script:
   ```bash
   python main.py
   ```

4. Open the Weave Dashboard:
   - After running the evaluation, navigate to your WandB dashboard.
   - Visualize the results and compare model performance.

---

## Key Benefits of Using Weave and WandB

1. **Custom Operations**: Define bespoke evaluation metrics as Weave "ops," enabling fine-tuned assessments.
2. **Seamless Integration**: Easily incorporate Weave with existing WandB workflows for better collaboration.
3. **Real-time Dashboards**: Monitor and compare LLM performance interactively using WandB’s visualization tools.
4. **Flexible Evaluation**: Evaluate any LLM by defining new models and datasets with minimal changes.

---

## Example Use Cases

- Comparing the contextual relevance of different LLMs (e.g., GPT-4 vs GPT-4-mini).
- Evaluating document-based question answering systems.
- Visualizing performance metrics of fine-tuned LLMs for specific tasks.


