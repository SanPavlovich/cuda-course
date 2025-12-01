#include <iostream>
using namespace std;


class matrix {
private:
    int** data;
    int nrows;
    int ncols;

    int** new_matrix(int nrows, int ncols) {
        int** A = new int*[nrows];
        for (int i=0; i < nrows; i++) {
            A[i] = new int[ncols];
        }
        return A;
    }

    void delete_matrix(int** A, int nrows) {
        for (int i=0; i < nrows; i++) {
            delete[] A[i];
        }
        delete[] A;
    }

    void swap(int& a, int& b) {
        int t = a;
        a = b;
        b = t;
        return;
    }

public:
    matrix(int nrows, int ncols) {
        cout << "matrix()" << endl;
        this->nrows = nrows;
        this->ncols = ncols;
        this->data = new_matrix(nrows, ncols);
    }

    // конструктор копирования, например: 
    // matrix A(2, 2);
    // martix B(A); здесь B будет равна А
    matrix(const matrix& other) {
        this->data = new_matrix(other.nrows, other.ncols);
        for (int i=0; i < other.nrows; i++) {
            for (int j=0; j < other.ncols; j++) {
                this->data[i][j] = other.data[i][j];
            }
        }
        this->nrows = other.nrows;
        this->ncols = other.ncols;
    }

    ~matrix() { 
        cout << "~matrix" << endl;
        delete_matrix(this->data, this->nrows);
    }

    // перегрузка оператора присваивания
    // matrix A(2, 2);
    // matrix B(3, 3);
    // B = A; в этом случае вызовется метод operator= для переменной B
    // B.operator=(A); запись выше эквивалентна этой записи
    // в переменную B положится ссылка объекта на самого себя
    matrix& operator= (const matrix& other) {
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

    matrix operator* (const matrix& other) {
        if (this->ncols != other.nrows) {
            throw std::invalid_argument("matrixes with this shapes can't be multiplied!");
        }
        matrix C(this->nrows, other.ncols);
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
        return C;
    }

    matrix operator+ (const matrix& other) {
        if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
            throw std::invalid_argument("matrixes with this shapes can't be added!");
        }
        matrix C(this->nrows, this->ncols);
        int sum;
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                C.data[i][j] = this->data[i][j] + other.data[i][j];
            }
        }
        return C;
    }

    matrix operator- (const matrix& other) {
        if ((this->nrows != other.nrows) || (this->ncols != other.ncols)) {
            throw std::invalid_argument("matrixes with this shapes can't be added!");
        }
        matrix C(this->nrows, this->ncols);
        int sum;
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                C.data[i][j] = this->data[i][j] - other.data[i][j];
            }
        }
        return C;
    }

    matrix& transpose() {
        // случай, когда матрица не квадратная и транспонирование приведет к изменению размерности
        if (this->nrows != this->ncols) {
            matrix A_T(this->ncols, this->nrows);
            for (int i=0; i < this->nrows; i++) {
                for (int j=0; j < this->ncols; j++) {
                    A_T.data[j][i] = this->data[i][j];
                }
            }
            delete_matrix(this->data, this->nrows);
            this->data = A_T.data;
            this->nrows = A_T.nrows;
            this->ncols = A_T.ncols;
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

    int get_nrows() {
        return this->nrows;
    }
    int get_ncols() {
        return this->ncols;
    }

    int** get_data_pointer() {
        //cout << "this->data: " << this->data << endl;
        return this->data;
    }

    void print() {
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                cout << this->data[i][j] << " ";
            }
            cout << endl;
        }
    }

    void rand_init() {
        for (int i=0; i < this->nrows; i++) {
            for (int j=0; j < this->ncols; j++) {
                this->data[i][j] = rand() % 10;
            }
        }
    }
};

int main() {
    matrix A(3, 2);
    A.rand_init();
    cout << "matrix A:" << endl;
    A.print();

    cout << "matrix A.T:" << endl;
    A.transpose();
    A.print();

    /*matrix B(2, 2);
    B.rand_init();
    cout << "matrix B:" << endl;
    B.print();

    matrix C = A + B;
    cout << "matrix C:" << endl;
    C.print();

    matrix D = A * B;
    cout << "matrix D:" << endl;
    D.print();*/

    return 0;
}