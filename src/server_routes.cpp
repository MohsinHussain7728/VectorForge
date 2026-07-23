#include "server_routes.h"

#include "json_utils.h"

#include <fstream>
#include <sstream>

void registerServerRoutes(
    httplib::Server &svr,
    VectorDB &db,
    DocumentDB &docDB,
    OllamaClient &ollama,
    int dims)
{
    // CORS preflight
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res); res.status = 204;
    });


    // GET /status
    svr.Get("/status", [&](const httplib::Request &, httplib::Response &res)
            {
        cors(res);
        bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"ollamaAvailable\":"  << (up ? "true" : "false")
           << ",\"embedModel\":"       << jS(ollama.embedModel)
           << ",\"genModel\":"         << jS(ollama.genModel)
           << ",\"docCount\":"         << docDB.size()
           << ",\"docDims\":"          << docDB.getDims()
           << ",\"demoDims\":"         << dims
           << ",\"demoCount\":"        << db.size() << '}';
        res.set_content(ss.str(), "application/json"); });

    svr.Get("/stats", [&](const httplib::Request &, httplib::Response &res)
            {
        cors(res);
        std::ostringstream ss;
        ss << "{\"count\":"      << db.size()
           << ",\"dims\":"       << dims
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        res.set_content(ss.str(), "application/json"); });

    // Serve index.html
    svr.Get("/", [](const httplib::Request &, httplib::Response &res)
            {
        std::ifstream f("index.html");
        if (!f.is_open()) { res.status = 404; return; }
        res.set_content(
            std::string(std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()),
            "text/html"); });
}