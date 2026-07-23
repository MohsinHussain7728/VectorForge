#include "chunker.h"
#include "ollama_client.h"
#include "json_utils.h"
#include "httplib.h"
#include "document_db.h"
#include "vector_item.h"
#include "bruteforce.h"
#include "kdtree.h"
#include "hnsw.h"
#include "vector_db.h"
#include "demo_data.h"
#include "demo_routes.h"
#include "document_routes.h"
#include "server_routes.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>

static const int DIMS = 16;   // demo vectors
// Doc embeddings dimension is determined at runtime from Ollama's model output

// =====================================================================
//  DATA TYPES
// =====================================================================




// =====================================================================
//  DISTANCE METRICS
// =====================================================================


// =====================================================================
//  BRUTE FORCE
// =====================================================================


// =====================================================================
//  KD-TREE
// =====================================================================



// =====================================================================
//  HNSW — Hierarchical Navigable Small World
// =====================================================================



// =====================================================================
//  VECTOR DATABASE  (demo 16D index)
// =====================================================================

// =====================================================================
//  JSON HELPERS
// =====================================================================

// =====================================================================
//  TEXT CHUNKER
// =====================================================================

// =====================================================================
//  OLLAMA CLIENT  — wraps local Ollama REST API
//  Install:  https://ollama.com
//  Models:   ollama pull nomic-embed-text
//            ollama pull llama3.2
// =====================================================================

// =====================================================================
//  DOCUMENT DATABASE  — HNSW over real Ollama embeddings
// =====================================================================

// =====================================================================
//  DEMO DATA  (16D categorical vectors)
// =====================================================================

// =====================================================================
//  HTTP SERVER
// =====================================================================

int main() {
    VectorDB   db(DIMS);
    DocumentDB docDB;
    OllamaClient ollama;

    loadDemo(db);

    // Check Ollama at startup (non-fatal)
    bool ollamaUp = ollama.isAvailable();
    std::cout << "=== VectorDB Engine ===" << std::endl;
    std::cout << "http://localhost:8080" << std::endl;
    std::cout << db.size() << " demo vectors | " << DIMS << " dims | HNSW+KD-Tree+BruteForce" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)") << std::endl;
    if (ollamaUp) std::cout << "  embed model: " << ollama.embedModel
                            << "  gen model: "   << ollama.genModel << std::endl;

    httplib::Server svr;

    registerDemoRoutes(svr, db, DIMS);

    registerDocumentRoutes(
    svr,
    docDB,
    ollama
    );

    registerServerRoutes(
    svr,
    db,
    docDB,
    ollama,
    DIMS
);

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    svr.listen("0.0.0.0", 8080);
    return 0;
}
