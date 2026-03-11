#include "chat.hpp"
#include<ctime>
#include<iostream>

void NLP::training::embedding_training(const int& train_count,str_vec dt,const int& embed_dim) {
	float lr = 0.01f;
	std::cout << "train start\n";
	srand(static_cast<unsigned int>(time(NULL)));

	for (int epoches = 0; epoches < train_count; epoches++) {
		lr = lr * 0.99f;
		for (size_t j = 0; j < dt.size(); j++) {

			str_vec data_tokens = ppcs.tokenize(dt[j]);
			if (data_tokens.empty()) continue;

			for (size_t p = 0; p < data_tokens.size(); p++) {

				if (ppcs.word2id.find(data_tokens[p]) == ppcs.word2id.end()) continue;

				int central_word = ppcs.word2id[data_tokens[p]];
				std::vector<int> right_word;

				if (p > 0 && ppcs.word2id.find(data_tokens[p - 1]) != ppcs.word2id.end())
					right_word.push_back(ppcs.word2id[data_tokens[p - 1]]);

				if (p + 1 < data_tokens.size() && ppcs.word2id.find(data_tokens[p + 1]) != ppcs.word2id.end())
					right_word.push_back(ppcs.word2id[data_tokens[p + 1]]);

				if (right_word.empty()) continue;

				int wrong_data[3];
				for (int r = 0; r < 3; r++) {
					do {
						wrong_data[r] = rand() % ppcs.word_id;
					} while (
						wrong_data[r] == central_word ||
						std::find(right_word.begin(), right_word.end(), wrong_data[r]) != right_word.end()
						);
				}

				for (size_t z = 0; z < right_word.size() + 3; z++) {
					int target_data;
					float right_score;

					if (z < right_word.size()) {
						target_data = right_word[z];
						right_score = 1.0f;
					}
					else {
						target_data = wrong_data[z - right_word.size()];
						right_score = 0.0f;
					}

					if (target_data >= ppcs.embedding_table.size()) continue;
					if (central_word >= ppcs.embedding_table.size()) continue;

					float score = similarityScore(ppcs.embedding_table[central_word], ppcs.embedding_table[target_data]);

					float error = right_score - score;
					flt_vec center_copy = ppcs.embedding_table[central_word];

					for (size_t q = 0; q < center_copy.size() && q < ppcs.embedding_table[target_data].size(); q++) {
						ppcs.embedding_table[central_word][q] += lr * error * ppcs.embedding_table[target_data][q];
						ppcs.embedding_table[target_data][q] += lr * error * center_copy[q];
					}
				}
			}
		}
		std::cout << epoches+1 << std::endl;
	}
	for (int i = 0; i < embed_dim; i++) {
		ppcs.embedding_table[ppcs.word2id["<PAD>"]][i] = 0;
	}
}