#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <algorithm>

#include "vector_item.h"
#include "distance_utils.h"

class HNSW
{
private:
    struct Node
    {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;

    int M;
    int M0;
    int ef_build;

    float mL;

    int topLayer = -1;
    int entryPt = -1;

    std::mt19937 rng;

    int randLevel();

    std::vector<std::pair<float, int>> searchLayer(
        const std::vector<float> &q,
        int ep,
        int ef,
        int lyr,
        DistFn dist);

    std::vector<int> selectNbrs(
        std::vector<std::pair<float, int>> &cands,
        int maxM);

public:
    HNSW(int m = 16, int efBuild = 200);

    void insert(
        const VectorItem &item,
        DistFn dist);

    std::vector<std::pair<float, int>> knn(
        const std::vector<float> &q,
        int k,
        int ef,
        DistFn dist);

    void remove(int id);

    struct GraphInfo
    {
        int topLayer;
        int nodeCount;

        std::vector<int> nodesPerLayer;
        std::vector<int> edgesPerLayer;

        struct NV
        {
            int id;
            std::string metadata;
            std::string category;
            int maxLyr;
        };

        struct EV
        {
            int src;
            int dst;
            int lyr;
        };

        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo();

    size_t size() const;
};