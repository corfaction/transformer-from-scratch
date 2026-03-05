#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <list>
#include <cstdint>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <json.hpp>

namespace py = pybind11;

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        std::hash<int> hasher;
        return hasher(p.first) ^ (hasher(p.second) << 1);
    }
};

class Tokenizer {
private:
    std::list<int> tokens;
    std::unordered_map<std::string, int> vocab;
    std::vector<std::pair<int, int>> merges;
    std::unordered_map<std::pair<int, int>, int, PairHash> pair_freq;
    std::vector<std::string> id_to_token;
public:
    Tokenizer() {
        vocab.reserve(256); 
        id_to_token.reserve(256);

        for (int i = 0; i < 256; ++i) {
            std::string token = std::string(1, static_cast<char>(i));
            vocab[token] = i;
            id_to_token.push_back(token);
        }
    }

    void train(std::string path, int tokens_size) {

        // Opening a data file

        std::ifstream file(path + "/dataset.txt");
        std::string content;
        if (file) {
            file.seekg(0, std::ios::end);
            size_t size = file.tellg();
            file.seekg(0, std::ios::beg);
        
            content.resize(size);
            file.read(&content[0], size);
        
            file.close();
        } else {std::cout << "Error opening file: " << path + "/dataset.txt" << std::endl; return;}

        // text to bytes conversion

        std::vector<uint8_t> bytes(content.begin(), content.end());

        // bytes to token conversion
        
        for(int b = 0; b < bytes.size(); ++b) {
            tokens.push_back(vocab.at(std::string(1, bytes[b])));
        }

        // initial count of token pair frequencies

        for(auto it = tokens.begin(); it != std::prev(tokens.end()); ++it) {
            std::pair<int,int> p = {*it, *std::next(it)};
            pair_freq[p]++;
        }

        std::pair<int,int> best_pair;
        int max_freq;

        for(int i = 0; vocab.size() < tokens_size; ++i) {

            // finding the most frequent pair

            max_freq = 0;
            for(auto &entry : pair_freq) {
                if (entry.second > max_freq) {
                    max_freq = entry.second;
                    best_pair = entry.first;
                }
            }

            // create new token

            std::string new_token =
                id_to_token[best_pair.first] +
                id_to_token[best_pair.second];

            vocab[new_token] = vocab.size();
            id_to_token.push_back(new_token);
            merges.push_back(best_pair);
            pair_freq.erase(best_pair);
            
            // replace the token pairs with a new token and recalculate the frequencies of the new pairs

            for (auto it = tokens.begin(); it != tokens.end(); ) {
                auto next = std::next(it);

                if (next == tokens.end()) break;

                if (*it == best_pair.first && *next == best_pair.second) {
                    auto prev = (it == tokens.begin()) ? tokens.end() : std::prev(it);
                    auto next_next = std::next(next);

                    if (prev != tokens.end())
                        if(--pair_freq[{*prev, *it}] <= 0)
                            pair_freq.erase({*prev, *it});

                    if (next_next != tokens.end())
                        if(--pair_freq[{*next, *next_next}] <= 0) 
                            pair_freq.erase({*next, *next_next});
                        
                    tokens.erase(next);
                    *it = vocab.size() - 1;

                    if (prev != tokens.end())
                        pair_freq[{*prev, *it}]++;

                    auto new_next = std::next(it);

                    if (new_next != tokens.end())
                        pair_freq[{*it, *new_next}]++;

                    ++it;

                } else {++it;}
            }
        }
    }

    // Export tokenizer

    void save(const std::string& path) {

        nlohmann::json vocab_json;

        for (size_t i = 0; i < id_to_token.size(); i++) {

            const std::string& token = id_to_token[i];

            std::vector<int> bytes;
            bytes.reserve(token.size());

            for (unsigned char c : token)
                bytes.push_back(c);

            vocab_json[std::to_string(i)] = bytes;
        }

        std::ofstream file_vocab(path + "/vocab.json");
        file_vocab << vocab_json.dump(2);
        file_vocab.close();


        std::ofstream file_merges(path + "/merges.txt");

        for (auto &p : merges) {
            file_merges << p.first << "," << p.second << "\n";
        }

        file_merges.close();
    }

    // import tokenizer

    void load(const std::string& path) {

        vocab.clear();
        id_to_token.clear();
        merges.clear();

        std::ifstream vocab_file(path + "/vocab.json");

        nlohmann::json vocab_json;
        vocab_file >> vocab_json;

        size_t vocab_size = vocab_json.size();

        id_to_token.resize(vocab_size);
        vocab.reserve(vocab_size);

        for (auto& [id_str, byte_array] : vocab_json.items()) {

            int id = std::stoi(id_str);

            std::string token;
            token.reserve(byte_array.size());

            for (auto& b : byte_array)
                token.push_back(static_cast<char>(b.get<int>()));

            vocab[token] = id;
            id_to_token[id] = token;
        }

        vocab_file.close();

        std::ifstream merges_file(path + "/merges.txt");

        std::string line;

        while (std::getline(merges_file, line)) {

            if (line.empty() || line[0] == '#')
                continue;

            std::istringstream iss(line);

            int a, b;
            char comma;

            if (iss >> a >> comma >> b)
                merges.push_back({a, b});
        }

        merges_file.close();
    }
};

PYBIND11_MODULE(tokenizer, m) {
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def("train", &Tokenizer::train)
        .def("save", &Tokenizer::save)
        .def("load", &Tokenizer::load);
}