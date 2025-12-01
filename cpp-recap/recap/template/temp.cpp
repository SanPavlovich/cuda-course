#include <iostream>
#include <vector>
using namespace std;

// статическая память - рассчитывается во время компиляции, сколько памяти потребуется программе для хранения данных в начале

// std::vector<double> vec(size); - динамически выделяет память под вектор нужного размера и очищает её, когда vec пропадает из поля видимости
// std::vector<double>* vec = new std::vector<double>(size); - а вот это костыльный и плохой способ. Контейнер vector итак делает эту работу за нас

/*vector<int> sum(const vector<int>& vec1, const vector<int>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("vec1 and vec2 size's must be equal!");
        // terminate called after throwing an instance of 'std::invalid_argument'
        // what():  vec1 and vec2 size's must be equal!
    }
    
    int result_dim = vec1.size(); 
    vector<int> vec_sum(result_dim);

    for (size_t i = 0; i < vec1.size(); i++) {
        vec_sum[i] = vec1[i] + vec2[i];
    }

    return vec_sum;
}*/

template<typename T>
void print(const vector<T>& vec) {
    for (const auto& elem : vec) {
        cout << elem << " ";
    }
    cout << endl;
}

template<typename T>
vector<T> sum(const vector<T>& vec1, const vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("vec1 and vec2 size's must be equal!");
        // terminate called after throwing an instance of 'std::invalid_argument'
        // what():  vec1 and vec2 size's must be equal!
    }
    
    int result_dim = vec1.size(); 
    vector<T> vec_sum(result_dim);

    for (size_t i = 0; i < vec1.size(); i++) {
        vec_sum[i] = vec1[i] + vec2[i];
    }

    return vec_sum;
}

int main() {
    vector<int> nums1 = {1, 2, 3, 4, 5};
    vector<int> nums2 = {1, 2, 3, 4, 5};
    vector<int> vector_sum;
    vector_sum = sum(nums1, nums2);
    print(vector_sum);

    vector<float> fnums1 = {1.1, 2.2, 3.4, 4.5, 5.0};
    vector<float> fnums2 = {1, 2, 3, 4, 5};
    vector<float> fvector_sum;
    fvector_sum = sum(fnums1, fnums2);
    print(fvector_sum);
    return 0;
}