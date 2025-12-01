#ifndef MATRIX_HPP
#define MATRIX_HPP

template <typename T>
class matrix {
private:
    T** data;
    size_t nrows;
    size_t ncols;

    T** new_matrix(size_t nrows, size_t ncols);

    void delete_matrix(T** A, size_t nrows);

    void swap(T& a, T& b);

    void swap_size(size_t& a, size_t& b);

public:
    matrix();

    matrix(size_t nrows, size_t ncols);

    matrix(const matrix& other);

    ~matrix();

    matrix& operator= (const matrix& other);

    matrix operator* (const matrix& other);

    matrix operator* (const matrix& other) const;

    matrix& operator*= (const matrix& other);

    matrix operator+ (const matrix& other);

    matrix operator+ (const matrix& other) const;

    matrix& operator+= (const matrix& other);

    matrix operator- (const matrix& other);

    matrix& transpose();

    int get_nrows();

    int get_ncols();

    int** get_data_pointer();

    void print();

    void rand_init();
};

#endif