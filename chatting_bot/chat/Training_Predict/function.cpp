#include"chat.hpp"
#include<cmath>
//#include<iostream>

float NLP::training::sigmoid(const float& x) {
	return 1.0f / (1.0f + exp(-x));
}

float NLP::training::similarityScore(const flt_vec& a, const flt_vec& b) {
	if (a.size() != b.size())return -9.0f;
	float sum = 0.0f;
	for (size_t i = 0; i < a.size(); i++) {
		sum += a[i] * b[i];
	}
	float score = NLP::training::sigmoid(sum);
	return score;
}

float NLP::training::cosine_similarity(flt_vec a, flt_vec b) {
    //내적 구하기
    float sum = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    //각 벡터 길이 계산
    float len_a = 0.0f, len_b = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        len_a += a[i] * a[i];
        len_b += b[i] * b[i];
    }
    len_a = sqrt(len_a);
    len_b = sqrt(len_b);

    if (len_a == 0 || len_b == 0) return 0.0f;
    return sum / (len_a * len_b);
}

flt_vec NLP::training::softmax(const flt_vec& scores) {
    //std::cout << "softmax" << std::endl;
    flt_vec probs(scores.size());
    //std::cout << 1 << std::endl;
    float max_score = scores[0];
    //std::cout << 1.1 << std::endl;
    for (float s : scores)
        if (s > max_score) max_score = s;

    //std::cout << 2 << std::endl;
    float sum = 0.0f;
    for (int i = 0; i < scores.size(); i++) {
        probs[i] = std::exp(scores[i] - max_score); // exp 유지
        sum += probs[i];
    }

    //std::cout << 3 << std::endl;
    for (int i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }

    //std::cout << 4 << std::endl;
    return probs;
}

std::vector<std::pair<std::string, float>> NLP::training::intent_softmax(const std::vector<std::pair<std::string, float>>& scores) {
    std::vector<std::pair<std::string, float>> probs(scores.size());

    // 1. max 값 찾기
    float max_score = scores[0].second;
    for (std::pair<std::string, float> s : scores)
        if (s.second > max_score) max_score = s.second;

    // 2. exp(score - max) 계산
    float sum = 0.0f;
    for (int i = 0; i < scores.size(); i++) {
        probs[i].second = std::exp(scores[i].second - max_score);
        probs[i].first = scores[i].first;
        sum += probs[i].second;
    }

    // 3. 정규화
    for (int i = 0; i < probs.size(); i++) {
        probs[i].second /= sum;
    }

    return probs;
}

flt_vec NLP::training::mat_vec_mul(const Matrix& W, const flt_vec& x) {
    flt_vec result(W.size(), 0.0f);
    for (int i = 0; i < W.size(); i++) {
        for (int j = 0; j < x.size(); j++) {
            result[i] += W[i][j] * x[j];
        }
    }
    return result;
}