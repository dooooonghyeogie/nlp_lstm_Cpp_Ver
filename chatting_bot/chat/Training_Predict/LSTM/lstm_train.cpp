#include"chat.hpp"
#include<iostream>

void NLP::training::lstm_train(
    const int& train_count,
    std::vector<std::pair<std::string, std::string>>& conversation,
    const int& embed_dim
) {
    float lr = 0.2f;

    for (int t = 0; t < train_count; t++) {
        //std::cout << "why?\n";
        float total_loss = 0.0f;
        //std::cout << "-1\n";
        for (int i = 0; i < conversation.size(); i++) {
            //std::cout << "0\n";
            lstm.reset_state();
            //std::cout << "1\n";
            std::string whole_converation = conversation[i].first + " " + conversation[i].second;
            str_vec data;
            data = ppcs.tokenize(whole_converation);
            if (data.size() < 2) continue;
            //std::cout << "2\n";
            float sentence_loss = 0;
            for (int j = 0; j < data.size() - 1; j++) {

                int input_id = ppcs.word2id[data[j]];
                int target_id = ppcs.word2id[data[j + 1]];
                flt_vec target(lstm.vocab_size, 0);
                target[target_id] = 1;
                if (ppcs.id2word[input_id] == "<pad>")continue;

                lstm.zero_grad();
                lstm.forward(ppcs.embedding_table[input_id]);
                lstm.probs = softmax(lstm.logits);

                lstm.backward(target);
                //std::cout << "data : " << j << "\n";
                float loss = -log(lstm.probs[target_id]);
                sentence_loss += loss;
                lstm.update(lr);


                for (int k = 0; k < embed_dim; k++) {
                    ppcs.embedding_table[input_id][k] -= lr * lstm.dx[k];
                }
            }
            std::fill(lstm.hidden.begin(), lstm.hidden.end(), 0.0f);
            std::fill(lstm.logits.begin(), lstm.logits.end(), 0.0f);
            std::fill(lstm.cell.begin(), lstm.cell.end(), 0.0f);
            sentence_loss /= (data.size() - 1);
            total_loss += sentence_loss;
            //std::cout << "data number : " << i << std::endl;
        }

        total_loss /= conversation.size();

        system("cls");
        std::cout << "epoches : " << t << " loss : " << total_loss << "\n\n";
        std::cout << lr << std::endl;
        lr *= 0.999f;
        if (t + 1 == train_count)break;
    }
    //std::cout << 1 << std::endl;
}