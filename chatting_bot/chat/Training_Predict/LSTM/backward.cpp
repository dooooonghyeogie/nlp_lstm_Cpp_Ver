#include"chat.hpp"
#include<iostream>

void NLP::training::LSTM::backward(const flt_vec& target)
{
    //std::cout << "backward" << std::endl;
    //std::cout << 1 << std::endl;
    // ---------- 1. dlogits = probs - target ----------
    dlogits.assign(vocab_size, 0.0f);
    for (int i = 0; i < vocab_size; i++) {
        dlogits[i] = probs[i] - target[i];
    }
    //std::cout << 2 << std::endl;
    // ---------- 2. dW_out, db_out ----------
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < hidden_size; j++)
            dW_out[i][j] += dlogits[i] * hidden[j];
        db_out[i] += dlogits[i];
    }
    //std::cout << 3 << std::endl;
    // ---------- 3. dh = W_out^T * dlogits ----------
    flt_vec dh(hidden_size, 0.0f);
    for (int j = 0; j < hidden_size; j++) {
        for (int i = 0; i < vocab_size; i++) {
            dh[j] += W_out[i][j] * dlogits[i];
        }
    }
    //std::cout << 4 << std::endl;
    // ---------- 4. do ----------
    flt_vec do_(hidden_size, 0.0f);
    for (int k = 0; k < hidden_size; k++) {
        float tanh_c = tanh(cell[k]);
        do_[k] = dh[k] * tanh_c * o[k] * (1 - o[k]);
    }
    //std::cout << 5 << std::endl;
    // ---------- 5. dc ----------
    flt_vec dc(hidden_size, 0.0f);
    for (int k = 0; k < hidden_size; k++) {
        float tanh_c = tanh(cell[k]);
        dc[k] = dh[k] * o[k] * (1 - tanh_c * tanh_c);
    }
    //std::cout << 6 << std::endl;
    // ---------- 6. di, df, dg ----------
    flt_vec di(hidden_size, 0.0f), df(hidden_size, 0.0f), dg(hidden_size, 0.0f);

    for (int k = 0; k < hidden_size; k++) {
        di[k] = dc[k] * g[k] * i[k] * (1 - i[k]);
        df[k] = dc[k] * c_prev[k] * f[k] * (1 - f[k]);
        dg[k] = dc[k] * i[k] * (1 - g[k] * g[k]);
    }
    //std::cout << 7 << std::endl;
    // ---------- 7. dWi, dUi, db ----------
    for (int j = 0; j < hidden_size; j++) {
        for (int i = 0; i < input_size; i++) {
            dWi[j][i] += di[j] * x[i];
            dWf[j][i] += df[j] * x[i];
            dWo[j][i] += do_[j] * x[i];
            dWg[j][i] += dg[j] * x[i];
        }
        //std::cout << 1;
        for (int i = 0; i < hidden_size; i++) {
            dUi[j][i] += di[j] * h_prev[i];
            dUf[j][i] += df[j] * h_prev[i];
            dUo[j][i] += do_[j] * h_prev[i];
            dUg[j][i] += dg[j] * h_prev[i];
        }
        //std::cout << 2;
        dbi[j] += di[j];
        dbf[j] += df[j];
        dbo[j] += do_[j];
        dbg[j] += dg[j];
    }
    //std::cout << 8 << std::endl;
    // ---------- 8. dx ----------
    dx.assign(input_size, 0.0f);
    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            dx[i] += W_i[j][i] * di[j];
            dx[i] += W_f[j][i] * df[j];
            dx[i] += W_o[j][i] * do_[j];
            dx[i] += W_g[j][i] * dg[j];
        }
    }
    //std::cout << 9 << std::endl;
    // ---------- 9. dh_prev ----------
    dh_prev.assign(hidden_size, 0.0f);
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            dh_prev[i] += U_i[j][i] * di[j];
            dh_prev[i] += U_f[j][i] * df[j];
            dh_prev[i] += U_o[j][i] * do_[j];
            dh_prev[i] += U_g[j][i] * dg[j];
        }
    }
    //std::cout << 10 << std::endl;
    // ---------- 10. dc_prev ----------
    dc_prev.assign(hidden_size, 0.0f);
    for (int k = 0; k < hidden_size; k++)
        dc_prev[k] = dc[k] * f[k];
}

void NLP::training::LSTM::zero_grad() {

    auto zero_matrix = [](Matrix& m) {
        for (auto& row : m)
            for (auto& v : row)
                v = 0.0f;
        };

    zero_matrix(dWi);
    zero_matrix(dWf);
    zero_matrix(dWo);
    zero_matrix(dWg);

    zero_matrix(dUi);
    zero_matrix(dUf);
    zero_matrix(dUo);
    zero_matrix(dUg);

    zero_matrix(dW_out);

    std::fill(dbi.begin(), dbi.end(), 0.0f);
    std::fill(dbf.begin(), dbf.end(), 0.0f);
    std::fill(dbo.begin(), dbo.end(), 0.0f);
    std::fill(dbg.begin(), dbg.end(), 0.0f);
    std::fill(db_out.begin(), db_out.end(), 0.0f);

    std::fill(dhidden.begin(), dhidden.end(), 0.0f);
    std::fill(dlogits.begin(), dlogits.end(), 0.0f);
}