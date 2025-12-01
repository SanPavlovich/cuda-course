#include <iostream>
using namespace std;

void swap(int& a, int& b) {
    int t = a;
    a = b;
    b = t;
    return;
}

int main() {
    /*int a = 5;
    int b = 4;
    cout << "a: " << a << ", " << "b: " << b << endl;
    swap(a, b);
    cout << "a: " << a << ", " << "b: " << b << endl;*/
    int array[] = {1, 2};
    cout << array[0] << " " << array[1];
    swap(array[0], array[1]);
    cout << array[0] << " " << array[1];
    return 0;
}