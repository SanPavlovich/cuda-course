#include <iostream>

class Point {
private:
    int x;
    int y;

public:
    Point(int value_x=0, int value_y=0) {
        x = value_x;
        y = value_y;
    }

    void print() {
        std::cout << "point x: " << x << std::endl;
        std::cout << "point y: " << y << std::endl;
    }
};

int main() {
    Point a(4, 5);
    a.print();
    return 0;
}