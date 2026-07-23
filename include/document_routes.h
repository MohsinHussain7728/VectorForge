#pragma once

#include "httplib.h"
#include "document_db.h"
#include "ollama_client.h"

void registerDocumentRoutes(
    httplib::Server& svr,
    DocumentDB& docDB,
    OllamaClient& ollama
);