#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <ctime>
#include<fstream>
#include<sstream>
#include<iostream>

typedef std::vector<std::string> str_vec;
typedef std::vector<float> flt_vec;
typedef std::vector<std::vector<float>> Matrix;

class NLP
{
public:
    class Preprocessor
    {
    public:
        int word_id = 0;
        str_vec id2word;
        std::unordered_map<std::string, int> word2id;
        std::vector<flt_vec> embedding_table;
        std::unordered_map<std::string, int> embed_tb;

        str_vec tokenize(const std::string& sentence);
        void make_vocab(const str_vec& tokens);
        void make_embedding_table(const int& embed_dim);
        flt_vec sentence_to_vector(const str_vec& tokens, int embed_dim);
    };

    class training
    {
    public:
        class LSTM {
        public:
            int input_size = 0;
            int hidden_size = 0;
            int vocab_size = 0;

            flt_vec hidden;
            flt_vec cell;
            flt_vec probs;

            Matrix W_i, W_f, W_o, W_g;
            Matrix U_i, U_f, U_o, U_g;
            flt_vec b_i, b_f, b_o, b_g;

            Matrix W_out;
            flt_vec b_out;
            flt_vec logits;

            float loss = 0;

            flt_vec x;
            flt_vec i, f, o, g;
            flt_vec c_prev;
            flt_vec h_prev;

            Matrix dWi, dWf, dWo, dWg;
            Matrix dUi, dUf, dUo, dUg;
            flt_vec dbi, dbf, dbo, dbg;
            Matrix dW_out;
            flt_vec db_out;

            flt_vec dx;
            flt_vec dhidden;
            flt_vec dlogits;

            flt_vec dc_prev;
            flt_vec dh_prev;

            // ===== 기본 생성자 추가 =====
            LSTM() {}

            // ===== 기존 생성자 =====
            LSTM(int input_size, int hidden_size, int vocab_size)
                : input_size(input_size), hidden_size(hidden_size), vocab_size(vocab_size)
            {
                float limit = sqrt(1.0f / input_size);
                std::mt19937 gen(std::random_device{}());
                std::uniform_real_distribution<float> dis(-limit, limit);

                auto init_matrix = [&](int rows, int cols) {
                    Matrix m(rows, flt_vec(cols));
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < cols; j++)
                            m[i][j] = dis(gen);
                    return m;
                    };

                auto init_dmatrix = [&](int rows, int cols) {
                    Matrix m(rows, flt_vec(cols));
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < cols; j++)
                            m[i][j] = 0;
                    return m;
                    };

                W_i = init_matrix(hidden_size, input_size);
                W_f = init_matrix(hidden_size, input_size);
                W_o = init_matrix(hidden_size, input_size);
                W_g = init_matrix(hidden_size, input_size);

                U_i = init_matrix(hidden_size, hidden_size);
                U_f = init_matrix(hidden_size, hidden_size);
                U_o = init_matrix(hidden_size, hidden_size);
                U_g = init_matrix(hidden_size, hidden_size);

                dWi = init_dmatrix(hidden_size, input_size);
                dWf = init_dmatrix(hidden_size, input_size);
                dWo = init_dmatrix(hidden_size, input_size);
                dWg = init_dmatrix(hidden_size, input_size);

                dUi = init_dmatrix(hidden_size, hidden_size);
                dUf = init_dmatrix(hidden_size, hidden_size);
                dUo = init_dmatrix(hidden_size, hidden_size);
                dUg = init_dmatrix(hidden_size, hidden_size);

                W_out = init_matrix(vocab_size, hidden_size);
                dW_out = init_dmatrix(vocab_size, hidden_size);

                logits.assign(vocab_size, 0.0f);

                hidden.assign(hidden_size, 0.0f);
                cell.assign(hidden_size, 0.0f);

                b_i.assign(hidden_size, 0.0f);
                b_o.assign(hidden_size, 0.0f);
                b_g.assign(hidden_size, 0.0f);
                b_f.assign(hidden_size, 1.0f);

                dbi.assign(hidden_size, 0.0f);
                dbo.assign(hidden_size, 0.0f);
                dbg.assign(hidden_size, 0.0f);
                dbf.assign(hidden_size, 0.0f);

                b_out.assign(vocab_size, 0.0f);
                db_out.assign(vocab_size, 0.0f);

                dlogits.assign(vocab_size, 0.0f);
                dx.assign(input_size, 0.0f);
            }

            void forward(const flt_vec& embed_table);
            void backward(const flt_vec& target);
            void zero_grad();
            void update(const float& lr);
            void reset_state();
        };

    private:
        Preprocessor ppcs;
        std::unordered_map<std::string, std::pair<flt_vec, float>> vector_of_Intent;
        str_vec intent_list;
        LSTM lstm;

    public:
        static float sigmoid(const float& x);
        static flt_vec mat_vec_mul(const Matrix& W, const flt_vec& x);
        float similarityScore(const flt_vec& a, const flt_vec& b);
        static float cosine_similarity(flt_vec a, flt_vec b);
        static flt_vec softmax(const flt_vec& scores);

        std::vector<std::pair<std::string, float>>
            intent_softmax(const std::vector<std::pair<std::string, float>>& scores);

        void embedding_training(const int& train_count, str_vec, const int& embed_dim);
        void lstm_train(
            const int& train_count,
            std::vector<std::pair<std::string, std::string>>& conversation,
            const int& embed_dim);
        void train(const int& train_count, const int& embed_dim);
        void make_commant(const std::string& commant);
    };

public:
    int embed_dim = 16;
    training trn;

    void model_train(const int& train_count) {
        trn.train(train_count, embed_dim);
    }

    void answer(std::string& commant) {
        trn.make_commant(commant);
    }
};

class Cbot {
public:
    std::unordered_map<std::string,int> word2id;
    str_vec id2word;
    std::vector<flt_vec> embedding_table;
private:
    NLP::training::LSTM lstm;
    Cbot() {
        std::fstream EmbedTb("C:\\Users\\이동혁\\Desktop\\코드\\chatting_bot\\chat\\Training_Predict\\trained_data\\embedding_table.txt");
        if (!EmbedTb.is_open()) {
            std::cout << -1 << std::endl;
        }
        std::cout << "model loading\n";
        std::string line;
        while (std::getline(EmbedTb,line)) {
            std::stringstream ss(line);
            std::string w, t;

            if (line.empty()) continue;
            std::getline(ss, w, '\t');
            std::getline(ss, t);
        }
    }
};