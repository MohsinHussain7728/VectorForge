#pragma once

#include <vector>
#include <functional>
#include <string>

using DistFn = std::function<float(
    const std::vector<float> &,
    const std::vector<float> &)>;

float cosine(const std::vector<float> &a,
             const std::vector<float> &b);

float euclidean(const std::vector<float> &a,
                const std::vector<float> &b);

float manhattan(const std::vector<float> &a,
                const std::vector<float> &b);

DistFn getDistFn(const std::string& m);