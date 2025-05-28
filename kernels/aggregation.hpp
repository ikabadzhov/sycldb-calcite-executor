#pragma once

#include <string>

template <typename T>
inline T element_operation(T a, T b, const std::string &op)
{
    if (op == "*")
        return a * b;
    else if (op == "/")
        return a / b;
    else if (op == "+")
        return a + b;
    else if (op == "-")
        return a - b;
    else
        return 0;
}

template <typename T>
void perform_operation(T result[], const T a[], const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b[i], op);
}

template <typename T>
void perform_operation(T result[], T a, const T b[], bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a, b[i], op);
}

template <typename T>
void perform_operation(T result[], const T a[], T b, bool flags[], int size, const std::string &op)
{
    for (int i = 0; i < size; i++)
        if (flags[i])
            result[i] = element_operation(a[i], b, op);
}

template <typename T, typename U>
void aggregate_operation(U &result, const T a[], bool flags[], int size, const std::string &op)
{
    if (op == "SUM")
    {
        result = 0;
        for (int i = 0; i < size; i++)
            if (flags[i])
                result += a[i];
    }
    else
    {
        // std::cout << "Unsupported aggregate operation: " << op << std::endl;
    }
}
