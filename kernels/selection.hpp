#pragma once

#include <sycl/sycl.hpp>

#include "common.hpp"

enum comp_op
{
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE
};
enum logical_op
{
    NONE,
    AND,
    OR
};

inline comp_op get_comp_op(std::string op)
{
    if (op == "==")
        return EQ;
    else if (op == ">=")
        return GE;
    else if (op == "<=")
        return LE;
    else if (op == "<")
        return LT;
    else if (op == ">")
        return GT;
    else if (op == "!=" || op == "<>")
        return NE;
    else
        return EQ;
}

inline logical_op get_logical_op(std::string op)
{
    if (op == "AND")
        return AND;
    else if (op == "OR")
        return OR;
    else
        return NONE;
}

// logicals are AND, OR etc. while comparisons are ==, <= etc.
// So checking alpha characters is enough to determine if the operation is logical.
inline bool is_filter_logical(const std::string &op)
{
    for (int i = 0; i < op.length(); i++)
        if (!isalpha(op[i]))
            return false;
    return true;
}

template <typename T>
inline bool compare(comp_op CO, T a, T b)
{
    switch (CO)
    {
    case EQ:
        return a == b;
    case NE:
        return a != b;
    case LT:
        return a < b;
    case LE:
        return a <= b;
    case GT:
        return a > b;
    case GE:
        return a >= b;
    default:
        return false;
    }
}

inline bool logical(logical_op logic, bool a, bool b)
{
    switch (logic)
    {
    case AND:
        return a && b;
    case OR:
        return a || b;
    case NONE:
        return b;
    default:
        return false;
    }
}

class EmptyKernel : public KernelDefinition
{
public:
    EmptyKernel(int len) : KernelDefinition(len) {}

    void operator()() const {}
};

class LogicalKernel : public KernelDefinition
{
private:
    logical_op logic;
    const bool *input_flags;
    const bool *local_flags;
    bool *output_flags;
public:
    LogicalKernel(logical_op log, const bool *input, const bool *local, bool *output, int len)
        : KernelDefinition(len), logic(log), input_flags(input), local_flags(local), output_flags(output)
    {}

    LogicalKernel(logical_op log, bool *f1, bool *f2, int len)
        : LogicalKernel(log, f1, f2, f1, len)
    {}

    void operator()(sycl::id<1> idx) const
    {
        output_flags[idx] = logical(logic, input_flags[idx], local_flags[idx]);
    }
};

class SelectionKernelColumns : public KernelDefinition
{
private:
    comp_op comparison;
    logical_op logic;
    const bool *input_flags;
    bool *output_flags;
    const int *operand1, *operand2;
public:
    SelectionKernelColumns(comp_op comp, logical_op log, const bool *input, bool *output, const int *op1, const int *op2, int len)
        : KernelDefinition(len), comparison(comp), logic(log), input_flags(input), output_flags(output), operand1(op1), operand2(op2)
    {}

    SelectionKernelColumns(comp_op comp, logical_op log, bool *f, const int *op1, const int *op2, int len)
        : SelectionKernelColumns(comp, log, f, f, op1, op2, len)
    {}

    SelectionKernelColumns(const bool *input, bool *output, const int *op1, const std::string &op, const int *op2, const std::string &parent_op, int len)
        : SelectionKernelColumns(get_comp_op(op), get_logical_op(parent_op), input, output, op1, op2, len)
    {}

    SelectionKernelColumns(bool *f, const int *op1, const std::string &op, const int *op2, const std::string &parent_op, int len)
        : SelectionKernelColumns(get_comp_op(op), get_logical_op(parent_op), f, f, op1, op2, len)
    {}

    void operator()(sycl::id<1> idx) const
    {
        output_flags[idx] = logical(logic, input_flags[idx], compare(comparison, operand1[idx], operand2[idx]));
    }
};

class SelectionKernelLiteral : public KernelDefinition
{
private:
    comp_op comparison;
    logical_op logic;
    const bool *input_flags;
    bool *output_flags;
    const int *operand1;
    int value;
public:
    SelectionKernelLiteral(comp_op comp, logical_op log, const bool *input, bool *output, const int *op1, int val, int len)
        : KernelDefinition(len), comparison(comp), logic(log), input_flags(input), output_flags(output), operand1(op1), value(val)
    {}

    SelectionKernelLiteral(comp_op comp, logical_op log, bool *f, const int *op1, int val, int len)
        : SelectionKernelLiteral(comp, log, f, f, op1, val, len)
    {}

    SelectionKernelLiteral(const bool *input, bool *output, const int *op1, const std::string &op, int val, const std::string &parent_op, int len)
        : SelectionKernelLiteral(get_comp_op(op), get_logical_op(parent_op), input, output, op1, val, len)
    {}

    SelectionKernelLiteral(bool *f, const int *op1, const std::string &op, int val, const std::string &parent_op, int len)
        : SelectionKernelLiteral(get_comp_op(op), get_logical_op(parent_op), f, f, op1, val, len)
    {}

    void operator()(sycl::id<1> idx) const
    {
        output_flags[idx] = logical(logic, input_flags[idx], compare(comparison, operand1[idx], value));
    }
};

inline sycl::event selection(
    bool flags[],
    const int arr[],
    std::string op,
    int value,
    std::string parent_op,
    int col_len,
    sycl::queue &queue,
    const std::vector<sycl::event> deps)
{
    comp_op comparison = get_comp_op(op);
    logical_op logic = get_logical_op(parent_op);
    SelectionKernelLiteral kernel(comparison, logic, flags, arr, value, col_len);

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(deps);
            cgh.parallel_for(
                kernel.get_col_len(),
                kernel
            );
        }
    );

    // std::cout << "Running selection with comparison: " << op << " and parent op " << parent_op << std::endl;
}

inline sycl::event selection(
    bool flags[],
    const int operand1[],
    std::string op,
    const int operand2[],
    std::string parent_op,
    int col_len,
    sycl::queue &queue,
    const std::vector<sycl::event> deps)
{
    comp_op comparison = get_comp_op(op);
    logical_op logic = get_logical_op(parent_op);
    SelectionKernelColumns kernel(comparison, logic, flags, operand1, operand2, col_len);

    return queue.submit(
        [&](sycl::handler &cgh)
        {
            cgh.depends_on(deps);
            cgh.parallel_for(
                kernel.get_col_len(),
                kernel
            );
        }
    );

    // std::cout << "Running selection with comparison: " << op << " and parent op " << parent_op << std::endl;
}
