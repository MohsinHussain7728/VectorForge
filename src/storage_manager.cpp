#include "storage_manager.h"

#include <sstream>
#include <vector>
#include <string>
#include <iostream>

StorageManager::StorageManager() : db(nullptr) {}

StorageManager::~StorageManager() {
    close();
}

bool StorageManager::open(const std::string& dbPath) {
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        std::cerr << "Failed to open database: "
                  << sqlite3_errmsg(db) << std::endl;

        sqlite3_close(db);
        db = nullptr;
        return false;
    }

    std::cout << "[StorageManager] Database opened successfully.\n";
    return true;
}

void StorageManager::close() {
    if (db != nullptr) {
        sqlite3_close(db);
        db = nullptr;

        std::cout << "[StorageManager] Database closed.\n";
    }
}

std::string StorageManager::serializeEmbedding(
    const std::vector<float>& embedding)
{
    std::ostringstream oss;

    for (size_t i = 0; i < embedding.size(); i++) {
        oss << embedding[i];

        if (i != embedding.size() - 1)
            oss << ",";
    }

    return oss.str();
}

std::vector<float> StorageManager::deserializeEmbedding(
    const std::string& embeddingStr)
{
    std::vector<float> embedding;

    std::stringstream ss(embeddingStr);
    std::string value;

    while (std::getline(ss, value, ',')) {
        embedding.push_back(std::stof(value));
    }

    return embedding;
}

bool StorageManager::saveVector(const VectorItem& item)
{
    const char* sql =
        "INSERT OR REPLACE INTO vectors(id, metadata, category, embedding) VALUES(?, ?, ?, ?);";

    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare INSERT statement.\n";
        return false;
    }

    sqlite3_bind_int(stmt, 1, item.id);

    sqlite3_bind_text(
        stmt,
        2,
        item.metadata.c_str(),
        -1,
        SQLITE_TRANSIENT
    );

    sqlite3_bind_text(
        stmt,
        3,
        item.category.c_str(),
        -1,
        SQLITE_TRANSIENT
    );

    std::string embedding = serializeEmbedding(item.emb);

    sqlite3_bind_text(
        stmt,
        4,
        embedding.c_str(),
        -1,
        SQLITE_TRANSIENT
    );

    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        std::cerr << "Failed to insert vector.\n";
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);

    return true;
}


std::vector<VectorItem> StorageManager::loadVectors()
{
    std::vector<VectorItem> vectors;

    const char* sql =
        "SELECT id, metadata, category, embedding FROM vectors;";

    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare SELECT statement.\n";
        return vectors;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        VectorItem item;

        item.id = sqlite3_column_int(stmt, 0);

        item.metadata =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

        item.category =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));

        std::string embedding =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));

        item.emb = deserializeEmbedding(embedding);

        vectors.push_back(item);
    }

    sqlite3_finalize(stmt);

    return vectors;
}


bool StorageManager::initializeSchema() {
    const char* createVectorsTable = R"(
        CREATE TABLE IF NOT EXISTS vectors(
            id INTEGER PRIMARY KEY,
            metadata TEXT,
            category TEXT,
            embedding TEXT
        );
    )";

    const char* createDocumentsTable = R"(
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding TEXT NOT NULL
        );
    )";

    char* errMsg = nullptr;

    if (sqlite3_exec(db, createVectorsTable, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Error creating vectors table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    if (sqlite3_exec(db, createDocumentsTable, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Error creating documents table: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    std::cout << "[StorageManager] Database schema initialized.\n";
    return true;
}

bool StorageManager::deleteVector(int id)
{
    const char* sql =
        "DELETE FROM vectors WHERE id = ?;";

    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::cerr << "Failed to prepare DELETE statement.\n";
        return false;
    }

    sqlite3_bind_int(stmt, 1, id);

    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        std::cerr << "Failed to delete vector.\n";
        sqlite3_finalize(stmt);
        return false;
    }

    sqlite3_finalize(stmt);

    return true;
}