#include "chat.hpp"
#include<iostream>

void NLP::training::make_commant(const std::string& commant) {
	str_vec tokens = ppcs.tokenize("<user> <sos> "+ commant + " <eos>");
	std::string answer;
	const int len = tokens.size();
	for (int i = 0;;i++) {
		//std::cout << i << std::endl;

		if (ppcs.word2id.find(tokens.at(i)) == ppcs.word2id.end()) tokens[i] = "<unk>";

		//std::cout << "A";
		flt_vec x = ppcs.embedding_table[ppcs.word2id[tokens[i]]];
		//std::cout << "B";
		lstm.forward(x);
		//std::cout << "C\n";
		flt_vec probs = softmax(lstm.logits);
		if (i >= len-1) {
			std::pair<int, float> next_word = { -1,0.0f };
			for (int j = 0; j < probs.size(); j++) {
				if (probs[j] > next_word.second) {
					next_word = { j,probs[j] };
				}
			}

			if (ppcs.id2word[next_word.first] == "<eos>")break;
			tokens.push_back(ppcs.id2word[next_word.first]);
			if (ppcs.id2word[next_word.first] == "<bot>" || ppcs.id2word[next_word.first] == "<sos>")continue;
			answer += (ppcs.id2word[next_word.first] + " ");
		}
	}
	std::cout <<"bot : " << answer << std::endl;
}