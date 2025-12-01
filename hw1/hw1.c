#include <stdio.h>
#include <math.h>

int main() {
    double a, b, c;
    printf("Введите коэффициенты a, b и c: ");
    if (scanf("%lf %lf %lf", &a, &b, &c) != 3) {
        printf("incorrect\n");
        return 1;
    }

    // Проверка на корректность уравнения
    if (a == 0 && b == 0 && c == 0) {
        printf("any\n");
        return 0;
    } else if (a == 0 && b == 0) {
        printf("incorrect\n");
        return 0;
    } else if (a == 0) { // Линейное уравнение bx + c = 0
        printf("%.2f\n", -c / b);
        return 0;
    }

    // Вычисление дискриминанта
    double D = b * b - 4 * a * c;

    if (D > 0) {
        // Два вещественных корня
        double root1 = (-b + sqrt(D)) / (2 * a);
        double root2 = (-b - sqrt(D)) / (2 * a);
        printf("%.2f + %.2f\n", root1, sqrt(D));
        printf("%.2f - %.2f\n", root2, sqrt(D));
    } else if (D == 0) {
        // Один вещественный корень
        double root = -b / (2 * a);
        printf("%.2f\n", root);
    } else {
        // Мнимые корни
        printf("imaginary\n");
    }

    return 0;
}