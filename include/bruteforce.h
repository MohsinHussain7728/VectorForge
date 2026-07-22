#pragma once

#include <vector>
#include <functional>

#include "vector_item.h"
#include "distance_utils.h"


class BruteForce
{
private:
    std::vector<VectorItem> items;

public:
    void insert(const VectorItem& item);

    void remove(int id);

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q,
        int k,
        std::function<float(
            const std::vector<float>&,
            const std::vector<float>&)> dist
    );
};