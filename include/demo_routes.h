#pragma once 
// Registers all VectorDB demo Endpoints

#include "httplib.h"
#include "vector_db.h"

void registerDemoRoutes(
    httplib::Server& svr,
    VectorDB& db,
    int dims
);