#include "chat.hpp"

void NLP::training::LSTM::reset_state() {
    std::fill(hidden.begin(), hidden.end(), 0.0f);
    std::fill(cell.begin(), cell.end(), 0.0f);
}

void NLP::training::LSTM::update(const float& lr)
{
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            W_i[i][j] -= lr * dWi[i][j];
            W_f[i][j] -= lr * dWf[i][j];
            W_o[i][j] -= lr * dWo[i][j];
            W_g[i][j] -= lr * dWg[i][j];
        }

        for (int j = 0; j < hidden_size; j++) {
            U_i[i][j] -= lr * dUi[i][j];
            U_f[i][j] -= lr * dUf[i][j];
            U_o[i][j] -= lr * dUo[i][j];
            U_g[i][j] -= lr * dUg[i][j];
        }

        b_i[i] -= lr * dbi[i];
        b_f[i] -= lr * dbf[i];
        b_o[i] -= lr * dbo[i];
        b_g[i] -= lr * dbg[i];
    }

    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < hidden_size; j++)
            W_out[i][j] -= lr * dW_out[i][j];
        b_out[i] -= lr * db_out[i];
    }
}