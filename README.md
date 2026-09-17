VectorForge — Build a Vector Database from Scratch in C++

A fully working modular Vector Database built from scratch in C++ with a web UI, REST API, SQLite persistence, document ingestion, and a RAG pipeline powered by Ollama.

Implements HNSW, KD-Tree, and Brute Force search algorithms side-by-side, allowing their performance to be compared using different distance metrics.

Built as an educational project to understand how modern vector databases work internally — from indexing and similarity search to persistence, embeddings, and Retrieval-Augmented Generation.

What This Project Does

Feature

Description

3 Search Algorithms

HNSW, KD-Tree, and Brute Force — run all three and compare speed

3 Distance Metrics

Cosine similarity, Euclidean distance, Manhattan distance

16D Demo Vectors

Semantic demo vectors across CS, Math, Food, and Sports

2D PCA Scatter Plot

Live visualization of semantic space

Real Document Embedding

Ollama embeds text using nomic-embed-text (768D)

Document Chunking

Long documents are automatically split into overlapping chunks

Document Semantic Search

Search documents using vector similarity

RAG Pipeline

Retrieve relevant document chunks and generate answers using llama3.2

SQLite Persistence

Vector data is persisted locally using SQLite

Startup Restoration

Persisted vectors are loaded and indexes are rebuilt on startup

Persistent Deletion

Deleted vectors are removed from memory and SQLite

Full REST API

Vector CRUD, search, benchmark, HNSW info, documents, RAG, and status

Web UI

Search, benchmarking, document ingestion, visualization, and AI question answering

Architecture

Overall Architecture

                         ┌──────────────────┐
                         │      Web UI      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     REST API     │
                         └────────┬─────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
          ┌─────────────┐                    ┌─────────────┐
          │  VectorDB   │                    │ DocumentDB  │
          └──────┬──────┘                    └──────┬──────┘
                 │                                  │
        ┌────────┼────────┐                   ┌─────┴─────┐
        ▼        ▼        ▼                   ▼           ▼
   BruteForce KD-Tree   HNSW             BruteForce     HNSW
        │        │        │                   │           │
        └────────┴────────┘                   │           │
                 │                            │           │
                 └──────────────┬─────────────┘
                                ▼
                       ┌─────────────────┐
                       │ StorageManager  │
                       └────────┬────────┘
                                │
                                ▼
                             SQLite
                                │
                                ▼
                     data/vectorforge.db

Vector Search Pipeline

Client / Web UI
      │
      ▼
   REST API
      │
      ▼
   VectorDB
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
 Brute Force      KD-Tree         HNSW
      │              │              │
      └──────────────┴──────────────┘
                     │
                     ▼
              Search Results

Persistent Vector Storage

VectorDB
   │
   ▼
StorageManager
   │
   ▼
SQLite
   │
   └── data/vectorforge.db

Application Restart
   │
   ▼
SQLite
   │
   ▼
Load persisted vectors
   │
   ▼
VectorDB
   │
   ├── Brute Force rebuilt
   ├── KD-Tree rebuilt
   └── HNSW rebuilt

Document + RAG Pipeline

Document
   │
   ▼
Chunking
   │
   ▼
Ollama
nomic-embed-text
   │
   ▼
768D Embedding
   │
   ▼
DocumentDB
   │
   ├── Brute Force
   └── HNSW
   │
   ▼
Semantic Retrieval
   │
   ▼
Retrieved Context
   │
   ▼
Ollama
llama3.2
   │
   ▼
Generated Answer

How It Works

Vector Database

VectorForge stores vectors together with metadata and category information.

For every vector, the system maintains:

VectorDB
 │
 ├── In-memory store
 │
 ├── Brute Force index
 │
 ├── KD-Tree index
 │
 └── HNSW index

When a vector is inserted:

Vector
  │
  ├── Store in memory
  │
  ├── Insert into Brute Force
  │
  ├── Insert into KD-Tree
  │
  ├── Insert into HNSW
  │
  └── Save to SQLite

When the application restarts:

SQLite
   │
   ▼
StorageManager
   │
   ▼
VectorDB
   │
   ├── Restore vectors
   ├── Rebuild Brute Force
   ├── Rebuild KD-Tree
   └── Rebuild HNSW

This allows vector data to survive server restarts.

HNSW

HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbor search algorithm based on a multilayer graph.

Nodes are organized into multiple layers:

Layer 2:       A -------- D
                \        /

Layer 1:     A ---- B ---- D ---- F
                \    |    /

Layer 0:   A -- B -- C -- D -- E -- F

Search starts from the upper layers and progressively moves toward the closest region.

The current implementation uses:

M = 16
efConstruction = 200
efSearch = 50

HNSW is particularly useful for high-dimensional embeddings such as the 768-dimensional document embeddings used in this project.

KD-Tree

KD-Tree is a binary space-partitioning data structure.

Each node splits the vector space using one dimension.

             Root
              │
       ┌──────┴──────┐
       │             │
    Left Region   Right Region
       │             │
    ┌──┴──┐       ┌──┴──┐
    ...   ...     ...   ...

KD-Trees can work well for lower-dimensional data.

However, their effectiveness decreases as dimensionality increases because of the curse of dimensionality.

This project uses:

Demo vectors       → 16D
Document embeddings → 768D

This provides a practical comparison between KD-Tree and HNSW across different dimensionalities.

Brute Force Search

Brute Force compares the query vector against every stored vector.

Query
 │
 ├── Compare with Vector 1
 ├── Compare with Vector 2
 ├── Compare with Vector 3
 ├── ...
 └── Compare with Vector N
          │
          ▼
     Sort by distance
          │
          ▼
       Top-K results

Approximate complexity:

O(N × D)

where:

N = number of vectors

D = vector dimensionality

Brute Force provides an exact baseline for comparing the other search algorithms.

Prerequisites

The following are required on Windows:

MSYS2 — GCC/G++ compiler

Git — repository management

Ollama — embedding and LLM runtime

PowerShell — included with Windows

SQLite — already included in the repository

You do not need to install SQLite separately.

SQLite is included under:

third_party/sqlite/
├── sqlite3.c
├── sqlite3.h
├── sqlite3ext.h
└── shell.c

Step-by-Step Setup — Windows

Step 1 — Install MSYS2

MSYS2 provides the GCC/G++ compiler required to build VectorForge.

1. Download MSYS2

Go to:

https://www.msys2.org

Download and install MSYS2.

The default installation path is recommended:

C:\msys64

2. Open MSYS2 UCRT64

From the Windows Start Menu, open:

MSYS2 UCRT64

3. Update MSYS2

Run:

pacman -Syu

If MSYS2 asks you to close the terminal, close it and reopen MSYS2 UCRT64.

Run the update again if required:

pacman -Syu

4. Install GCC/G++

Run:

pacman -S mingw-w64-ucrt-x86_64-gcc

5. Add GCC to Windows PATH

Press:

Win + R

Enter:

sysdm.cpl

Then:

Advanced
   ↓
Environment Variables
   ↓
System Variables
   ↓
Path
   ↓
Edit
   ↓
New

Add:

C:\msys64\ucrt64\bin

Click OK on all windows.

6. Verify Installation

Open a new PowerShell window.

Run:

g++ --version

Also verify:

gcc --version

Step 2 — Install Git

Git is required to clone the VectorForge repository.

Download Git for Windows:

https://git-scm.com/download/win

Install it using the default settings.

Verify:

git --version

Example:

git version 2.x.x

Step 3 — Install Ollama

Ollama is used for:

Generating document embeddings

Generating RAG answers

Download Ollama for Windows:

https://ollama.com

Install Ollama.

After installation, Ollama normally runs in the background.

Step 3.1 — Install the Embedding Model

Open PowerShell:

ollama pull nomic-embed-text

VectorForge uses this model to convert documents and questions into vectors.

Expected embedding size:

768 dimensions

Step 3.2 — Install the LLM

Pull the language model:

ollama pull llama3.2

This model is used by the RAG pipeline to generate answers using retrieved document context.

Step 3.3 — Verify Ollama

Run:

ollama list

You should see:

nomic-embed-text
llama3.2

If Ollama is not already running, start it with:

ollama serve

Ollama Models Used

Model

Purpose

nomic-embed-text

Converts text into 768D embeddings

llama3.2

Generates answers for the RAG pipeline

Step 4 — Clone the Repository

Open PowerShell.

Clone the repository:

git clone https://github.com/YOUR_USERNAME/VectorForge.git

Move into the project:

cd VectorForge

Replace YOUR_USERNAME with your actual GitHub username.

Step 5 — Verify Project Structure

After cloning, the project should look approximately like:

VectorForge/
│
├── main.cpp
├── index.html
├── README.md
│
├── include/
│   ├── bruteforce.h
│   ├── chunker.h
│   ├── demo_data.h
│   ├── demo_routes.h
│   ├── distance_utils.h
│   ├── document_db.h
│   ├── document_routes.h
│   ├── hnsw.h
│   ├── httplib.h
│   ├── json_utils.h
│   ├── kdtree.h
│   ├── ollama_client.h
│   ├── server_routes.h
│   ├── storage_manager.h
│   ├── vector_db.h
│   └── vector_item.h
│
├── src/
│   ├── bruteforce.cpp
│   ├── chunker.cpp
│   ├── demo_data.cpp
│   ├── demo_routes.cpp
│   ├── distance_utils.cpp
│   ├── document_db.cpp
│   ├── document_routes.cpp
│   ├── hnsw.cpp
│   ├── json_utils.cpp
│   ├── kdtree.cpp
│   ├── ollama_client.cpp
│   ├── server_routes.cpp
│   ├── storage_manager.cpp
│   └── vector_db.cpp
│
├── data/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
└── third_party/
    └── sqlite/
        ├── sqlite3.c
        ├── sqlite3.h
        ├── sqlite3ext.h
        └── shell.c

Step 6 — Compile SQLite

VectorForge uses a vendored copy of SQLite.

From the project root, run:

gcc -c third_party/sqlite/sqlite3.c `
-Ithird_party/sqlite `
-o sqlite3.o

This creates:

sqlite3.o

in the project root.

Step 7 — Compile VectorForge

Compile the complete C++ project:

g++ -std=c++17 -O2 `
main.cpp src/*.cpp sqlite3.o `
-Iinclude -Ithird_party/sqlite `
-o db.exe -lws2_32

Command Explanation

-std=c++17

Uses C++17.

-O2

Enables compiler optimizations.

main.cpp src/*.cpp

Compiles the modular VectorForge backend.

sqlite3.o

Links the SQLite implementation.

-Iinclude

Adds the project header directory.

-Ithird_party/sqlite

Adds the SQLite header directory.

-lws2_32

Links Windows Winsock support required by the HTTP server.

After successful compilation:

db.exe

will be created.

Step 8 — Run VectorForge

You need Ollama and VectorForge running.

Terminal 1 — Start Ollama

If Ollama is not already running:

ollama serve

Keep this terminal open.

If Ollama is already running in the Windows system tray, you can skip this step.

Terminal 2 — Start VectorForge

Open another PowerShell window.

Go to the project:

cd VectorForge

Run:

.\db.exe

You should see something similar to:

=== VectorDB Engine ===
http://localhost:8080
20+ vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2

The exact vector count depends on the contents of your SQLite database.

Step 9 — Open the Web UI

Open your browser:

http://localhost:8080

The VectorForge web interface should now be available.

Using the Application

Tab 1 — Search

The Search tab works with the 16-dimensional demo vectors.

Try concepts such as:

binary tree
sushi
basketball
calculus

Choose Algorithm

HNSW
KD-Tree
Brute Force

Choose Distance Metric

Cosine
Euclidean
Manhattan

Click:

⚡ SEARCH

to perform the search.

You can also click:

▶ COMPARE ALL ALGOS

to compare the latency of all three search algorithms.

PCA Visualization

The scatter plot provides a 2D visualization of the demo vector space.

The demo vectors represent semantic categories such as:

CS
Math
Food
Sports

The visualization provides an intuitive way to see how vectors occupy different regions of the embedding space.

Tab 2 — Documents

The Documents tab allows you to insert real text and generate embeddings using Ollama.

Workflow

Text
 │
 ▼
Chunking
 │
 ▼
Ollama
 │
 ▼
768D Embedding
 │
 ▼
DocumentDB
 │
 ├── HNSW
 └── Brute Force

Steps

Enter a document title.

Example:

Operating Systems Notes

Paste your document text.

Click:

⚡ EMBED & INSERT

VectorForge splits long documents into overlapping chunks.

Each chunk receives its own 768-dimensional embedding.

The chunks are inserted into DocumentDB.

Current chunking configuration:

Chunk size  = 250 words
Overlap     = 30 words

Tab 3 — Ask AI

The Ask AI tab provides the RAG pipeline.

First insert documents using the Documents tab.

Then enter a question.

Example:

What is dynamic programming?

Click:

🤖 ASK AI

RAG Pipeline

Behind the scenes:

1. User Question
        │
        ▼
2. nomic-embed-text
        │
        ▼
3. 768D Query Vector
        │
        ▼
4. DocumentDB Search
        │
        ▼
5. Retrieve Relevant Chunks
        │
        ▼
6. Build Context Prompt
        │
        ▼
7. llama3.2
        │
        ▼
8. Generated Answer

The retrieved document chunks are provided to the LLM as context.

The UI also displays the retrieved context so you can see which document chunks were used.

SQLite Persistence

One of the major features added in Sprint 2 is persistent storage.

VectorForge stores vector data in:

data/vectorforge.db

The database stores information such as:

ID
Metadata
Category
Embedding

The SQLite schema is:

CREATE TABLE vectors (
    id INTEGER PRIMARY KEY,
    metadata TEXT,
    category TEXT,
    embedding TEXT
);

Vector Insert Persistence

When a vector is inserted:

POST /insert
      │
      ▼
   VectorDB
      │
      ├── In-memory store
      ├── Brute Force
      ├── KD-Tree
      └── HNSW
      │
      ▼
StorageManager
      │
      ▼
SQLite

Startup Restoration

When VectorForge starts:

Application Start
       │
       ▼
Open SQLite
       │
       ▼
Initialize Schema
       │
       ▼
Load Persisted Vectors
       │
       ▼
VectorDB
       │
       ├── Brute Force rebuilt
       ├── KD-Tree rebuilt
       └── HNSW rebuilt

The next available vector ID is also restored.

For example:

Before restart:
1
2
3
...
21

After restart:

nextId = 22

The next inserted vector receives:

ID = 22

Persistent Delete

Deleting a vector removes it from:

VectorDB memory
     │
     ├── Brute Force
     ├── KD-Tree
     └── HNSW

SQLite

This keeps the persistent database and in-memory indexes synchronized.

REST API Reference

The server exposes a REST API at:

http://localhost:8080

Demo Vector Endpoints

Method

Endpoint

Description

GET

/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw

K-NN vector search

POST

/insert

Insert a vector

DELETE

/delete/:id

Delete a vector by ID

GET

/items

List all vectors

GET

/benchmark?v=...&k=5&metric=cosine

Compare all three algorithms

GET

/hnsw-info

HNSW graph structure and layer statistics

GET

/stats

Database statistics

Document & RAG Endpoints

Method

Endpoint

Body

Description

POST

/doc/insert

{"title":"...","text":"..."}

Embed and store document

GET

/doc/list

—

List stored document chunks

DELETE

/doc/delete/:id

—

Delete document chunk

POST

/doc/search

{"question":"...","k":3}

Semantic document search

POST

/doc/ask

{"question":"...","k":3}

RAG: retrieve + generate

GET

/status

—

Ollama status and model information

Example API Requests

Search via curl

curl.exe "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"

Insert a Vector

curl.exe -X POST http://localhost:8080/insert `
  -H "Content-Type: application/json" `
  -d '{"metadata":"Test vector","category":"cs","embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,0.3,0.4,0.5,0.6,0.7]}'

Delete a Vector

curl.exe -X DELETE http://localhost:8080/delete/21

Insert a Document

curl.exe -X POST http://localhost:8080/doc/insert `
  -H "Content-Type: application/json" `
  -d '{"title":"Operating Systems","text":"Operating systems manage hardware resources and provide services to applications."}'

Ask a Question

curl.exe -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is an operating system?","k":3}'

Project Structure

VectorForge/
│
├── main.cpp
├── index.html
├── README.md
│
├── include/
│   ├── bruteforce.h
│   ├── chunker.h
│   ├── demo_data.h
│   ├── demo_routes.h
│   ├── distance_utils.h
│   ├── document_db.h
│   ├── document_routes.h
│   ├── hnsw.h
│   ├── httplib.h
│   ├── json_utils.h
│   ├── kdtree.h
│   ├── ollama_client.h
│   ├── server_routes.h
│   ├── storage_manager.h
│   ├── vector_db.h
│   └── vector_item.h
│
├── src/
│   ├── bruteforce.cpp
│   ├── chunker.cpp
│   ├── demo_data.cpp
│   ├── demo_routes.cpp
│   ├── distance_utils.cpp
│   ├── document_db.cpp
│   ├── document_routes.cpp
│   ├── hnsw.cpp
│   ├── json_utils.cpp
│   ├── kdtree.cpp
│   ├── ollama_client.cpp
│   ├── server_routes.cpp
│   ├── storage_manager.cpp
│   └── vector_db.cpp
│
├── data/
│   └── .gitkeep
│
├── uploads/
│   └── .gitkeep
│
└── third_party/
    └── sqlite/
        ├── sqlite3.c
        ├── sqlite3.h
        ├── sqlite3ext.h
        └── shell.c

Core Components

Component

Responsibility

VectorDB

Main vector database and index coordination

BruteForce

Exact nearest-neighbor search

KDTree

Exact spatial search for lower-dimensional vectors

HNSW

Approximate nearest-neighbor graph search

StorageManager

SQLite connection, schema, persistence, loading, and deletion

DocumentDB

Stores document chunks and performs semantic retrieval

OllamaClient

Communicates with Ollama for embeddings and generation

Document Routes

Document ingestion, search, deletion, and RAG

Server Routes

Vector search, insert, delete, benchmark, statistics, and HNSW APIs

Chunker

Splits documents into overlapping text chunks

Web UI

Search, visualization, benchmarking, documents, and RAG interface

Algorithm Comparison

Algorithm

Type

Main Characteristics

Brute Force

Exact

Simple baseline and exact results

KD-Tree

Exact

Spatial partitioning, useful for lower-dimensional data

HNSW

Approximate

Graph-based search, useful for high-dimensional embeddings

Brute Force

Complexity ≈ O(N × D)

Every vector is compared with the query.

KD-Tree

Uses recursive space partitioning and pruning.

HNSW

Uses a multilayer graph to navigate through the vector space.

Distance Metrics

VectorForge supports three distance functions.

Cosine Distance

Measures the angular difference between vectors.

Useful for:

Text embeddings
Semantic similarity
Document retrieval
RAG

Euclidean Distance

Measures straight-line distance.

d(x,y) = √Σ(xᵢ-yᵢ)²

Manhattan Distance

Measures the sum of absolute coordinate differences.

d(x,y) = Σ|xᵢ-yᵢ|

Common Issues

Problem

Solution

g++: command not found

Add C:\msys64\ucrt64\bin to Windows PATH

gcc: command not found

Add the MSYS2 UCRT64 bin directory to PATH

SQLite linker errors

Recompile sqlite3.o

undefined reference to WSA...

Make sure -lws2_32 is included

Ollama: OFFLINE

Start Ollama with ollama serve

Embedding takes a long time

Ollama may be downloading/loading the model

Model not found

Run ollama list and ollama pull <model>

Browser cannot connect

Make sure db.exe is running

Port 8080 already in use

Find and terminate the process using port 8080

RAG response is slow

Local LLM inference depends on available CPU/GPU resources

Vector data disappears after restart

Check that data/vectorforge.db exists and SQLite was included during compilation

Port 8080 Already in Use

Check which process is using port 8080:

netstat -ano | findstr :8080

You may see:

TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    12345

Terminate the process:

taskkill /PID 12345 /F

Replace 12345 with the actual PID.

Using a Smaller LLM

If llama3.2 is too slow, you can use a smaller model.

Pull:

ollama pull llama3.2:1b

Then update the generation model in:

include/ollama_client.h

Change:

std::string genModel = "llama3.2";

to:

std::string genModel = "llama3.2:1b";

Recompile:

g++ -std=c++17 -O2 main.cpp src/*.cpp sqlite3.o `
-Iinclude -Ithird_party/sqlite `
-o db.exe -lws2_32

Then restart VectorForge.
