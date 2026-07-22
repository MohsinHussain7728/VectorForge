#include "chunker.h"
#include "ollama_client.h"
#include "json_utils.h"
#include "httplib.h"
#include "document_db.h"
#include "vector_item.h"
#include "bruteforce.h"
#include "kdtree.h"
#include "hnsw.h"
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

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        for (auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
};

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

void loadDemo(VectorDB& db) {
    auto dist = getDistFn("cosine");
    // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    db.insert("Linked List: nodes connected by pointers", "cs",
        {0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
    db.insert("Binary Search Tree: O(log n) search and insert", "cs",
        {0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
    db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
        {0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
    db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
        {0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
    db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
        {0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
    db.insert("Calculus: derivatives integrals and limits", "math",
        {0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
    db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
        {0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
    db.insert("Probability: distributions random variables Bayes theorem", "math",
        {0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
    db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
        {0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
    db.insert("Combinatorics: permutations combinations generating functions", "math",
        {0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
    db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
    db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
        {0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
    db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
        {0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
    db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
        {0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
    db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
        {0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
    db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
        {0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
    db.insert("Football: tackles touchdowns field goals and strategy", "sports",
        {0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
    db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
        {0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
    db.insert("Chess: openings endgames tactics strategic board game", "sports",
        {0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
    db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
        {0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
}

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

    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });

    // ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto algo   = req.get_param_value("algo");   if (algo.empty())   algo   = "hnsw";

        auto out = db.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i) ss << ',';
            auto& h = out.hits[i];
            ss << "{\"id\":"        << h.id
               << ",\"metadata\":"  << jS(h.meta)
               << ",\"category\":"  << jS(h.cat)
               << ",\"distance\":"  << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":"       << jS(out.algo)
           << ",\"metric\":"     << jS(out.metric) << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || (int)emb.size() != DIMS) {
            res.set_content("{\"error\":\"invalid body\"}", "application/json"); return;
        }
        int id = db.insert(meta, cat, emb, getDistFn("cosine"));
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = db.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto items = db.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < items.size(); i++) {
            if (i) ss << ',';
            auto& v = items[i];
            ss << "{\"id\":"        << v.id
               << ",\"metadata\":"  << jS(v.metadata)
               << ",\"category\":"  << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto q = parseVec(req.get_param_value("v"));
        if ((int)q.size() != DIMS) {
            res.set_content("{\"error\":\"need " + std::to_string(DIMS) + "D vector\"}",
                            "application/json"); return;
        }
        int k = 5; try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric"); if (metric.empty()) metric = "cosine";
        auto b = db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":"       << b.hnswUs << ",\"itemCount\":" << b.n << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto gi = db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i) ss << ','; ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); i++) {
            if (i) ss << ',';
            auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); i++) {
            if (i) ss << ',';
            auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────

    // POST /doc/insert  {"title":"...","text":"..."}
    // Chunks the text, embeds each chunk via Ollama, stores in DocumentDB
    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto title = extractStr(req.body, "title");
        auto text  = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            res.set_content("{\"error\":\"need title and text\"}", "application/json"); return;
        }

        auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;

        for (int i = 0; i < (int)chunks.size(); i++) {
            auto emb = ollama.embed(chunks[i]);
            if (emb.empty()) {
                res.set_content(
                    "{\"error\":\"Ollama unavailable. "
                    "Install from https://ollama.com then run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2\"}",
                    "application/json");
                return;
            }
            std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i+1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            ids.push_back(docDB.insert(chunkTitle, chunks[i], emb));
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); i++) { if (i) ss << ','; ss << ids[i]; }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":"    << docDB.getDims() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // DELETE /doc/delete/123
    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int id  = std::stoi(req.matches[1]);
        bool ok = docDB.remove(id);
        res.set_content("{\"ok\":" + std::string(ok ? "true" : "false") + "}",
                        "application/json");
    });

    // GET /doc/list
    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        auto docs = docDB.all();
        std::ostringstream ss; ss << '[';
        for (size_t i = 0; i < docs.size(); i++) {
            if (i) ss << ',';
            // Truncate text preview to 120 chars
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "…";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << jS(docs[i].title)
               << ",\"preview\":" << jS(preview)
               << ",\"words\":"  << (int)std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1
               << '}';
        }
        ss << ']';
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/search {"question":"...","k":3}
    // Fast retrieval for the UI visualizer
    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        auto hits = docDB.search(qEmb, k);

        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        res.set_content(ss.str(), "application/json");
    });

    // POST /doc/ask  {"question":"...","k":3}
    // Full RAG pipeline: embed → retrieve → generate
    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        auto question = extractStr(req.body, "question");
        int  k        = extractInt(req.body, "k", 3);
        if (question.empty()) {
            res.set_content("{\"error\":\"need question\"}", "application/json"); return;
        }

        // Step 1: embed the question
        auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            res.set_content("{\"error\":\"Ollama unavailable\"}", "application/json"); return;
        }

        // Step 2: retrieve top-k relevant chunks
        auto hits = docDB.search(qEmb, k);

        // Step 3: build prompt
        std::ostringstream ctx;
        for (int i = 0; i < (int)hits.size(); i++) {
            ctx << "[" << (i+1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }
        std::string prompt =
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            "Just answer the question naturally.\n\n"
            "Context:\n" + ctx.str() +
            "Question: " + question + "\n\n"
            "Answer:";

        // Step 4: generate answer
        auto answer = ollama.generate(prompt);

        // Step 5: return everything
        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":"  << jS(ollama.genModel)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); i++) {
            if (i) ss << ',';
            ss << "{\"id\":"       << hits[i].second.id
               << ",\"title\":"    << jS(hits[i].second.title)
               << ",\"text\":"     << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    // GET /status
    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << jS(ollama.embedModel)
           << ",\"genModel\":"         << jS(ollama.genModel)
           << ",\"docCount\":"         << docDB.size()
           << ",\"docDims\":"          << docDB.getDims()
           << ",\"demoDims\":"         << DIMS
           << ",\"demoCount\":"        << db.size() << '}';
        res.set_content(ss.str(), "application/json");
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        cors(res);
        std::ostringstream ss;
        ss << "{\"count\":"      << db.size()
           << ",\"dims\":"       << DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json");
    });

    // Serve index.html
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html");
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
