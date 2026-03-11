#include "chat.hpp"
#include<cmath>

void NLP::training::LSTM::forward(const flt_vec& embed_table) {

    i.assign(hidden_size,0), f.assign(hidden_size,0), g.assign(hidden_size,0), o.assign(hidden_size,0);
    h_prev = hidden;
    c_prev = cell;
    x = embed_table;

    flt_vec Wx_i = mat_vec_mul(W_i, embed_table);
    flt_vec Wx_f = mat_vec_mul(W_f, embed_table);
    flt_vec Wx_o = mat_vec_mul(W_o, embed_table);
    flt_vec Wx_g = mat_vec_mul(W_g, embed_table);

    flt_vec Uh_i = mat_vec_mul(U_i, hidden);
    flt_vec Uh_f = mat_vec_mul(U_f, hidden);
    flt_vec Uh_o = mat_vec_mul(U_o, hidden);
    flt_vec Uh_g = mat_vec_mul(U_g, hidden);

    for (int k = 0; k < hidden_size; k++) {
        i[k] = sigmoid(Wx_i[k] + Uh_i[k] + b_i[k]);
        f[k] = sigmoid(Wx_f[k] + Uh_f[k] + b_f[k]);
        o[k] = sigmoid(Wx_o[k] + Uh_o[k] + b_o[k]);
        g[k] = tanh(Wx_g[k] + Uh_g[k] + b_g[k]);
    }

    for (int k = 0; k < hidden_size; k++) {
        cell[k] = f[k] * c_prev[k] + i[k] * g[k];
        hidden[k] = o[k] * tanh(cell[k]);
    }

    logits.assign(vocab_size, 0.0f);
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            logits[i] += W_out[i][j] * hidden[j];
        }
        logits[i] += b_out[i];
    }

    
}