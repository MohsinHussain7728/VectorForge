#pragma once

#include <vector>
#include <string>
#include "vector_item.h"
#include <sqlite3.h>

class StorageManager {
private:
    sqlite3* db;

    std::string serializeEmbedding(const std::vector<float>& embedding);
    std::vector<float> deserializeEmbedding(const std::string& embeddingStr);

public:
    StorageManager();
    ~StorageManager();

    bool open(const std::string& dbPath);
    void close();

    bool initializeSchema();

    bool saveVector(const VectorItem& item);
    std::vector<VectorItem> loadVectors();
    bool deleteVector(int id);
};