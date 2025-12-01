#include <iostream>
#include "point.hpp"
#include "point.cpp" //важно было включить этот файл, поскольку используется шаблонный класс и на этапе линковки main.exe вылетают ошибки

int main() {
    Point<int> p(2, 2);
    p.print();
    int sum = p.sum();
    cout << "sum: " << sum << endl;

    Point<float> pf(2.2, 3.9);
    pf.print();
    float fsum = pf.sum();
    cout << "fsum: " << fsum;

    return 0;
}