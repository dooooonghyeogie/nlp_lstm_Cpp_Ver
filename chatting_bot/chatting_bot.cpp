// chatting_bot.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include "chatting_bot.h"
#include"chat/chat.hpp"
#include<sstream>
#include<string>
#include<Windows.h>

using namespace std;

int main()
{
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	cout.tie(nullptr);
	NLP nlp;
	cout << "train_start\n";
	nlp.model_train(1000);
	cout << "finished\n";
	system("cls");
	cout << "train finishes start chat!!\n";
	string s;

	for (;;) {
		
		getline(std::cin,s);
		nlp.answer(s);
	}

	return 0;
}
