#include <iostream>
using namespace std;

void explain_pointer() {
    int a = 10;
    int* pa = &a;
    cout << "a: " << a << endl;
    cout << "pa: " << pa << endl;
    cout << "*pa: " << *pa << endl; //*pa - переход по адресу в памяти pa, в консоли выведется 10
    *pa = 20;
    cout << "changed a: " << a << endl;
    pa++; //адрес в памяти сместился на 4 байта: 0x8b07ffa64 --> 0x8b07ffa68
    cout << "pa: " << pa << endl;
    int* pp; //можно объявить указатель без инициализации
    return;
}

void explain_link() {
    int a = 10;
    int& aref = a;
    cout << "a: " << a << endl;
    cout << "aref: " << aref << endl;
    cout << "&aref: " << &aref << endl;
    // cout << "*aref: " << *aref << endl; - так работать не будет, ведь у ссылок нет оператора разыменования! Они сразу работают со значением, на которое они указывают
    // у указателей также есть своя арифметика, но для ссылок такое не работает.
    //int& pp; нельзя объявить ссылку без инициализации
    return;
}

void array_pointer(int* parr, size_t size) {
    for(int i = 0; i < size; i++) {
        parr[i] += 1;
    }
    //for(int i : parr) {
    //    cout << i;
    //}
}

void array_link(int& parr, size_t size) {
    for(int i = 0; i < size; i++) {
        //parr[i] += 1;
        return;
    }
}

void print(int* parr, size_t size) {
    for(int i = 0; i < size; i++) {
        cout << parr[i] << " ";
    }
    cout << endl;
}

int main() {
    //explain_pointer();
    //explain_link();

    int array[5] = {1, 2, 3, 4, 5};
    size_t size = sizeof(array) / sizeof(array[0]);
    print(array, size);
    array_pointer(array, size);
    print(array, size);
    return 0;
}