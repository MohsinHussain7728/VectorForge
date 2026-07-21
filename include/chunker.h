#pragma once

#include <string>
#include <vector>

std::vector<std::string> chunkText(
    const std::string& text,
    int chunkWords = 250,
    int overlapWords = 30);