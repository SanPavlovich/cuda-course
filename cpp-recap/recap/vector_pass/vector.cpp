#include <iostream>
#include <vector>
using namespace std;

void func(vector<int> vec) {
    vec.push_back(8);
    for (int elem : vec) {
        cout << elem << " ";
    }
    return;
}

void func_link(vector<int>& vec) {
    vec.push_back(8);
    for (int elem : vec) {
        cout << elem << " ";
    }
    return;
}

void func_pointer(vector<int>* ptr) {
    // ptr->size() == (*ptr).size()
    ptr->push_back(8);
    (*ptr).push_back(8);
    cout << "vector size: " << (*ptr).size() << endl;
    for (int i=0; i < ptr->size(); i++) {
        cout << ptr->at(i) << " ";
    }
    return;
}

void print(const vector<int>& vec) {
    //▎Преимущества использования константных ссылок:
    // 1. Эффективность: Вместо копирования объекта передается ссылка на него, что экономит время и память.
    // 2. Безопасность: Объект не может быть изменен внутри функции, так как ссылка является константной.

    for (int i : vec) {
        cout << i << " ";
    }
}

int main() {
    vector<int> v = {1, 2, 3, 4, 5};
    //func(v);
    //func_link(v);
    func_pointer(&v);
    cout << endl;
    print(v);
    /*for (auto elem : v) {
        cout << elem << " ";
    }*/
    return 0;
    
}