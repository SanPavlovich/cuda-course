#ifndef POINT_HPP
#define POINT_HPP

template<typename T>
class Point {
public:
    Point();
    Point(T x, T y);
    Point operator+ (const Point& other);
    T sum();
    void print();
private:
    T x;
    T y;
};

#endif