#pragma once

#include "httplib.h"
#include "vector_db.h"
#include "document_db.h"
#include "ollama_client.h"

void registerServerRoutes(
    httplib::Server& svr,
    VectorDB& db,
    DocumentDB& docDB,
    OllamaClient& ollama,
    int dims
);