# Lightweight Local RAG System with .NET

A **lightweight, local Retrieval-Augmented Generation (RAG)** system built with **.NET**, designed to prompt a locally hosted LLM (**phi3:mini**) using context from local `.txt`, `.md`, or `.pdf` files. Utilizes **Microsoft.Extensions.AI** and a local embedding model (**nomic-embed-text**) for fast, private, and offline text retrieval and generation.

---

## Features and parameters

- Local prompting: **phi3:mini** LLM
- Context extraction from `.txt`, `.md`, and `.pdf` files
- Local Embedding-based retrieval: **nomic-embed-text**
- Vector search: **cosine similarity**
- Chunk size (words): 300
- Chunk overlap: None
- Hashing algorithm to track file changes: SHA256

---

## Prerequisites

- **.NET 8.0+**
- **Ollama** installed locally

---

## NuGet Packages

```bash
dotnet add package OllamaSharp
dotnet add package PdfPig

