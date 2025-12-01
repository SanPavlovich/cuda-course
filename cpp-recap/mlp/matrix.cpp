#include <iostream>
#include "matrix.hpp"
using namespace std;

template<typename T>
T** matrix<T>::new_matrix(size_t nrows, size_t ncols) {
    T** A = new T*[nrows];
    for (int i=0; i < nrows; i++) {
        A[i] = new T[ncols];
    }
    return A;
}

template<typename T>
void matrix<T>::delete_matrix(T** A, size_t nrows) {
    if (A == nullptr)
        return;
    for (int i=0; i < nrows; i++) {
        delete[] A[i];
    }
    delete[] A;
};

template<typename T>
void matrix<T>::swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
    return;
}

template<typename T>
void matrix<T>::swap_size(size_t& a, size_t& b) {
    size_t t = a;
    a = b;
    b = t;
    return;
}

    
template<typename T>
matrix<T>::matrix() : data(nullptr), nrows(0), ncols(0) {
    cout << "matrix()" << endl;
}

template<typename T>
matrix<T>::matrix(size_t nrows, size_t ncols) {
    cout << "matrix(size_t nrows, size_t ncols)" << endl;
    this->nrows = nrows;
    this->ncols = ncols;
    this->data = new_matrix(nrows, ncols);
}

// конструктор копирования, например: 
// matrix A(2, 2);
// martix B(A); здесь B будет равна А
template<typename T>
matrix<T>::matrix(const matrix& other) {
    cout << "matrix(const matrix& other)"<< endl;
    this->data = new_matrix(other.nrows, other.ncols);
    for (int i=0; i < other.nrows; i++) {
        for (int j=0; j < other.ncols; j++) {
            this->data[i][j] = other.data[i][j];
        }
    }
    this->nrows = other.nrows;
    this->ncols = other.ncols;
}

template<typename T>
matrix<T>::~matrix() { 
    cout << "~matrix" << endl;
    delete_matrix(this->data, this->nrows);
}

// перегрузка оператора присваивания
// matrix A(2, 2);
// matrix B(3, 3);
// B = A; в этом случае вызовется метод operator= для переменной B
// B.operator=(A); запись выше эквивалентна этой записи
// в переменную B положится ссылка объекта на самого себя
template<typename T>
matrix<T>& matrix<T>::operator= (const matrix& other) {
    if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
        delete_matrix(this->data, this->nrows);
        this->data = new_matrix(other.nrows, other.ncols);
    }
    this->nrows = other.nrows;
    this->ncols = other.ncols;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            this->data[i][j] = other.data[i][j];
        }
    }
    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator* (const matrix& other) {
    if (this->ncols != other.nrows) {
        throw std::invalid_argument("matrixes with this shapes can't be multiplied!");
    }
    matrix<T> C(this->nrows, other.ncols);
    int sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < other.nrows; j++) {
            sum = 0;
            for (int k=0; k < this->ncols; k++) {
                sum += this->data[i][k] * other.data[k][j];
            }
            C.data[i][j] = sum;
        }
    }
    return C;
}

// эта функция гарантирует, что оператор *this не изменится благодаря слову const!
template<typename T>
matrix<T> matrix<T>::operator* (const matrix& other) const {
    if (this->ncols != other.nrows) {
        throw std::invalid_argument("matrixes with this shapes can't be multiplied!");
    }
    matrix<T> C(this->nrows, other.ncols);
    int sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < other.nrows; j++) {
            sum = 0;
            for (int k=0; k < this->ncols; k++) {
                sum += this->data[i][k] * other.data[k][j];
            }
            C.data[i][j] = sum;
        }
    }
    return C;
}

template<typename T>
matrix<T>& matrix<T>::operator*= (const matrix& other) {
    if (this->ncols != other.nrows) {
        throw std::invalid_argument("matrixes with this shapes can't be multiplied!");
    }
    /*matrix C(this->nrows, other.ncols);
    int sum;
    for (int i=0; i < this->ncols; i++) {
        for (int j=0; j < other.nrows; j++) {
            sum = 0;
            for (int k=0; k < ncols; k++) {
                sum += this->data[i][k] * other.data[k][j];
            }
            C.data[i][j] = sum;
        }
    }
    delete_matrix(this->data, this->nrows);
    this->data = C.data;
    this->nrows = C.nrows;
    this->ncols = C.ncols;*/
    T** C = new_matrix(this->nrows, other.ncols);
    T sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < other.ncols; j++) {
            sum = 0;
            for (int k=0; k < this->ncols; k++) {
                sum += this->data[i][k] * other.data[k][j];
            }
            C[i][j] = sum;
        }
    }
    delete_matrix(this->data, this->nrows);
    this->data = C;
    this->ncols = other.ncols;
    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator+ (const matrix& other) {
    cout << "operator+" << endl;
    if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
        throw std::invalid_argument("matrixes with this shapes can't be added!");
    }
    matrix<T> C(this->nrows, this->ncols);
    int sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            C.data[i][j] = this->data[i][j] + other.data[i][j];
        }
    }
    return C;
}

template<typename T>
matrix<T> matrix<T>::operator+ (const matrix& other) const {
    cout << "operator+ const" << endl;
    if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
        throw std::invalid_argument("matrixes with this shapes can't be added!");
    }
    matrix<T> C(this->nrows, this->ncols);
    int sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            C.data[i][j] = this->data[i][j] + other.data[i][j];
        }
    }
    return C;
}

template<typename T>
matrix<T>& matrix<T>::operator+= (const matrix& other) {
    cout << "operator+=" << endl;
    if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
        throw std::invalid_argument("matrixes with this shapes can't be added!");
    }
    
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            this->data[i][j] += other.data[i][j];
        }
    }
    return *this;
}

template<typename T>
matrix<T> matrix<T>::operator- (const matrix& other) {
    if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
        throw std::invalid_argument("matrixes with this shapes can't be added!");
    }
    matrix<T> C(this->nrows, this->ncols);
    int sum;
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            C.data[i][j] = this->data[i][j] - other.data[i][j];
        }
    }
    return C;
}

template<typename T>
matrix<T>& matrix<T>::transpose() {
    // случай, когда матрица не квадратная и транспонирование приведет к изменению размерности
    if (this->nrows != this->ncols) {
        /*matrix A_T(this->ncols, this->nrows);
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                A_T.data[j][i] = this->data[i][j];
            }
        }
        delete_matrix(this->data, this->nrows);
        this->data = A_T.data;
        this->nrows = A_T.nrows;
        this->ncols = A_T.ncols;*/

        T** A_T = new_matrix(this->ncols, this->nrows);
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                A_T[j][i] = this->data[i][j];
            }
        }
        delete_matrix(this->data, this->nrows);
        this->data = A_T;
        swap_size(this->nrows, this->ncols);
    }
    //случай, когда матрица квадратная и мы можем просто сделать swap(a[i][j], a[j][i])
    else {
        for (int i=0; i < this->nrows; i++) {
            for (int j=i+1; j < this->ncols; j++) {
                swap(this->data[i][j], this->data[j][i]);
            }
        }
    }
    return *this;
}

template<typename T>
int matrix<T>::get_nrows() {
    return this->nrows;
}

template<typename T>
int matrix<T>::get_ncols() {
    return this->ncols;
}

template<typename T>
int** matrix<T>::get_data_pointer() {
    //cout << "this->data: " << this->data << endl;
    return this->data;
}

template<typename T>
void matrix<T>::print() {
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            cout << this->data[i][j] << " ";
        }
        cout << endl;
    }
}

template<typename T>
void matrix<T>::rand_init() {
    for (int i=0; i < this->nrows; i++) {
        for (int j=0; j < this->ncols; j++) {
            this->data[i][j] = (rand() % 10);
        }
    }
}