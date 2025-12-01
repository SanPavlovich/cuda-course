#include <iostream>
using namespace std;

int** matmul(int** A, int** B, int nrows, int ncols) {
    int** C = new int*[nrows];
    for (int i = 0; i < nrows; i++) {
        C[i] = new int[nrows];
    }

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            int sum = 0;
            for (int k = 0; k < ncols; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

void rand_init(int** matrix, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            matrix[i][j]  = rand() % 10;
        }
    }
}

void print(int** matrix, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

// пример корректного возвращения ссылки
// если оставить int& max(int a, int b), то ссылка будет указывать на уничтоженный объект a или b, то есть на None
// ведь объекты int a, int b это временные копии и после завершения функции будут уничтожены
int& max(int& a, int& b)
{
    if (a > b) 
        return a;
    else
        return b;
}

// двумерный массив в с++ - это массив указателей на одномерные массивы
// то есть по сути int** A, где А указывает на массив указателей на 1-мерные массивы
// получается А --> [int* a1, ..., int* aN], где *aN = [elem1, ..., elemM], и тогда A.shape = (N, M) 

int main() {
    //int nrows = 5;
    //int ncols = 5;
    int nrows;
    int ncols;
    cout << "nrows: ", cin >> nrows;
    cout << "ncols: ", cin >> ncols;
    cout << "matrix shape: (" << nrows << ", " << ncols << ")" << endl;
    // выделение памяти под 1-d массив: int* A = new int[10];
    // очистка памяти:                  delete[] A;
    // обнуление указателя:             A = nullptr;
    int** A = new int*[nrows];
    for (int i = 0; i < nrows; i++) {
        A[i] = new int[ncols];
    }

    int** B = new int*[ncols];
    for (int i = 0; i < ncols; i++) {
        B[i] = new int[nrows];
    }

    rand_init(A, nrows, ncols);
    rand_init(B, ncols, nrows);
    cout << "matrix A:" << endl;
    print(A, nrows, ncols);
    cout << "matrix B:" << endl;
    print(B, ncols, nrows);

    int** C;
    C = matmul(A, B, nrows, ncols);
    cout << "matrix C:" << endl;
    print(C, nrows, nrows);

    for (int i = 0; i < nrows; i++) {
        delete[] C[i];
    }
    delete[] C;
    
    for (int i = 0; i < nrows; i++) {
        delete[] B[i];
    }
    delete[] B;
    
    for (int i = 0; i < ncols; i++) {
        delete[] A[i];
    }
    delete[] A;

    return 0;
}