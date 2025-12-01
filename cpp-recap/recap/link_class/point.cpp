#include <iostream>
#include "point.hpp"
using namespace std;

template<typename T>
Point<T>::Point() {
    this->x = 0;
    this->y = 0;
}

template<typename T>
Point<T>::Point(T x, T y) {
    this->x = x;
    this->y = y;
}

template<typename T>
Point<T> Point<T>::operator+ (const Point& other) {
    T new_x = this->x + other.x;
    T new_y = this->y + other.y;
    Point<T> p(new_x, new_y);
    return p;
}

template<typename T>
T Point<T>::sum() {
    T sum = this->x + this->y;
    return sum;
}

template<typename T>
void Point<T>::print() {
    cout << "x: " << x << ", " << "y: " << y << endl;
}