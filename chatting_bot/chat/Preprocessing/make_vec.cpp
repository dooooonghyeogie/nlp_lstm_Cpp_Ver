#include "chat.hpp"
#include <sstream>
#include <cctype>
#include <random>

void NLP::Preprocessor::make_embedding_table(const int& embed_dim)
{
    std::random_device rd;  // 시드값 생성
    std::mt19937 gen(rd()); // 난수 엔진
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < word_id; i++)
    {
        flt_vec table;
        for (int j = 0; j < 16; j++)
        {

            table.push_back(dis(gen));
        }
        embedding_table.push_back(table);
    }
}

flt_vec NLP::Preprocessor::sentence_to_vector(const str_vec& tokens, int embed_dim) {
    flt_vec vec(embed_dim, 0.0f);

    for (auto& token : tokens) {
        if (word2id.find(token) == word2id.end()) continue;
        int id = word2id[token];

        for (int i = 0; i < embed_dim; i++) {
            vec[i] += embedding_table[id][i];
        }
    }

    for (int i = 0; i < embed_dim; i++) {
        vec[i] /= tokens.size();
    }

    return vec;
}