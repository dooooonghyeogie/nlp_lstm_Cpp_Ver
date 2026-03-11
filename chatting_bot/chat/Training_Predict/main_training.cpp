#include "chat.hpp"
#include<fstream>
#include<iostream>
#include<sstream>
#include<Windows.h>



void NLP::training::train(const int& train_count, const int& embed_dim) {
    
	std::fstream data_file("C:\\Users\\이동혁\\Desktop\\코드\\chatting_bot\\chat\\train_data\\practice_data.txt");
    if (!data_file.is_open()) {
        std::cout << "파일 열기 실패\n";
        return;
    }
    str_vec special_tokens = {"<sos>","<eos>","<pad>","<user>","<bot>","<unk>"};//unk = 모르는 단어
    ppcs.make_vocab(special_tokens);

    str_vec sentences;
    std::vector<std::pair<std::string, std::string>>conversation;
    std::string line;
    while (std::getline(data_file, line)) {
        
        std::stringstream ss(line);
        std::string input, answer;

        if (line.empty()) continue;
        std::getline(ss, input, '\t');   // 앞부분
        std::getline(ss, answer);
        if (input.empty() || answer.empty()) continue;
        input = "<user> <sos> " + input + " <eos>";
        answer = "<bot> <sos> " + answer + " <eos>";
        sentences.push_back(input);
        sentences.push_back(answer);

        str_vec tokens = ppcs.tokenize(input);
        ppcs.make_vocab(tokens);
        tokens = ppcs.tokenize(answer);
        ppcs.make_vocab(tokens);
        conversation.push_back({ input,answer });
    }
    ppcs.make_embedding_table(embed_dim);

    std::cout << "word : " << ppcs.word_id + 1<<"\n";

    std::cout << "embed_training\n";
    embedding_training(train_count, sentences, embed_dim);
    std::cout << "embed_training_finished\nLSTM train start\n";

    lstm = LSTM(embed_dim, 64, ppcs.word2id.size());
    lstm_train(train_count, conversation, embed_dim);
    data_file.close();

    std::cout << "save embedding table\n";
    std::ofstream embed_File("C:\\Users\\이동혁\\Desktop\\코드\\chatting_bot\\chat\\Training_Predict\\trained_data\\embedding_table.txt", std::ios::app | std::ios::trunc | std::ios::trunc);
    

    if (embed_File.is_open()) {
        embed_File.clear();
        int k = 0;
        std::cout << ppcs.embedding_table.size() <<" "<<ppcs.id2word.size()<< " "<<ppcs.word_id<< std::endl;
        for (flt_vec&i : ppcs.embedding_table) {
            std::cout << k<<std::endl;
            embed_File << ppcs.id2word[k]<<" : ";
            for (const float& j : i) {
                embed_File << j << " ";
            }
            embed_File << std::endl;
            k++;
        }
        embed_File.close(); // 파일 닫기
    }
    std::cout << "saved\n";

    std::cout << "save weights and bias\n";
    std::ofstream weight_File("C:\\Users\\이동혁\\Desktop\\코드\\chatting_bot\\chat\\Training_Predict\\trained_data\\lstm_weight_bias.txt", std::ios::app | std::ios::trunc);
    

    if (weight_File.is_open()) {
        auto write_matrix = [&](const std::string& name, const Matrix& mat) {
            weight_File << name << "\n";
            for (const flt_vec& row : mat) {
                for (const float& v : row) {
                    weight_File << v << " ";
                }
                weight_File << "\n";
            }
            weight_File << "\n";
            };

        auto write_vector = [&](const std::string& name, const flt_vec& vec) {
            weight_File << name << "\n";
            for (const auto& v : vec) {
                weight_File << v << " ";
            }
            weight_File << "\n\n";
            };

        // Weight matrices
        write_matrix("W_out", lstm.W_out);

        write_matrix("W_i", lstm.W_i);
        write_matrix("W_f", lstm.W_f);
        write_matrix("W_o", lstm.W_o);
        write_matrix("W_g", lstm.W_g);

        write_matrix("U_i", lstm.U_i);
        write_matrix("U_f", lstm.U_f);
        write_matrix("U_o", lstm.U_o);
        write_matrix("U_g", lstm.U_g);

        // Bias vectors
        write_vector("b_out", lstm.b_out);

        write_vector("b_i", lstm.b_i);
        write_vector("b_f", lstm.b_f);
        write_vector("b_o", lstm.b_o);
        write_vector("b_g", lstm.b_g);

        weight_File.close();
    }
    std::cout << "saved\n";

    std::cout << "train finished\n";
}