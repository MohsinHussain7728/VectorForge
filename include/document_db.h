#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "hnsw.h"
#include "bruteforce.h"
#include "vector_item.h"
#include "distance_utils.h"

struct VectorItem;

// =====================================================================
// DOCUMENT ITEM
// =====================================================================


struct DocItem{

    int id;

    std::string title;

    std::string text;

    std::vector<float> emb;

    // Metadata
    std::string sourceType;      // txt, pdf, md, html, ...
    std::string originalFile;    // Original uploaded filename
    std::string createdAt;       // ISO-8601 timestamp

};


// =====================================================================
// DOCUMENT DATABASE
// =====================================================================


class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW hnsw;
    BruteForce bf;       // brute force fallback for small sets

    std::mutex mu;

    int nextId = 1;
    int dims   = 0;      // determined from first inserted embedding

public:
    DocumentDB();

    // Insert one chunk with its pre-computed embedding
    int insert(
        const std::string& title,
        const std::string& text,
        const std::vector<float>& emb
    );

    // Semantic search — returns top-k most similar chunks
    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q,
        int k,
        float max_dist = 0.7f
    );

    bool remove(int id);

    std::vector<DocItem> all();

    size_t size();

    int getDims();
};