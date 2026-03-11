#include "chat.hpp"
#include <sstream>
#include <cctype>

str_vec NLP::Preprocessor::tokenize(const std::string& sentence)
{
    std::vector<std::string> tokens;

    std::stringstream ss(sentence);
    std::string word;
    while (ss >> word)
    {
        for (char& c : word)
            c = std::tolower(static_cast<unsigned char>(c));
        tokens.push_back(word);
    }
    return tokens;
}

void NLP::Preprocessor::make_vocab(const str_vec& tokens)
{
    for (int i = 0; i < tokens.size(); i++)
    {
        if (word2id.find(tokens[i]) != word2id.end())
        {
            continue;
        }
        else
        {
            word2id[tokens[i]] = word_id;
            id2word.push_back(tokens[i]);
            word_id++;
        }
    }
    return;
}