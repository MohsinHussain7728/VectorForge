#pragma once

#include "httplib.h"

#include <string>
#include <vector>

std::string jS(const std::string& s);

std::string jVec(const std::vector<float>& v);

std::vector<float> parseVec(const std::string& s);

std::string extractStr(const std::string& body,
                       const std::string& key);

int extractInt(const std::string& body,
               const std::string& key,
               int def = 0);

bool parseBody(const std::string& b,
               std::string& meta,
               std::string& cat,
               std::vector<float>& emb);

void cors(httplib::Response& res);