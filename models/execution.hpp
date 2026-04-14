#pragma once

#include "../kernels/jit_kernels_impl.hpp"
#include <vector>
#include <memory>
#include <optional>
#include <hipSYCL/sycl/jit.hpp>
#include "../kernels/common.hpp"
#include "../kernels/selection.hpp"
#include "../kernels/projection.hpp"
#include "../kernels/aggregation.hpp"
#include "../kernels/join.hpp"
#include "../kernels/jit_kernels.hpp"

namespace acpp_jit = sycl::AdaptiveCpp_jit;

enum class KernelType : uint8_t
{
    EmptyKernel,
    LogicalKernel,
    SelectionKernelColumns,
    SelectionKernelLiteral,
    FillKernel,
    PerformOperationKernelColumns,
    PerformOperationKernelLiteralFirst,
    PerformOperationKernelLiteralSecond,
    BuildKeysHTKernel,
    FilterJoinKernel,
    BuildKeyValsHTKernel,
    FullJoinKernel,
    AggregateOperationKernel,
    GroupByAggregateKernel,
};

using JITOp = acpp_jit::dynamic_function_definition<void, sycl::item<1>, SYCLDBContext, bool&, uint64_t&>;

class KernelData
{
private:
    KernelType kernel_type;
    std::shared_ptr<KernelDefinition> kernel_def;
public:
    KernelData(KernelType kt, KernelDefinition *kd)
        : kernel_type(kt), kernel_def(std::shared_ptr<KernelDefinition>(kd))
    {}

    bool is_valid() const { return kernel_def != nullptr; }
    KernelType get_type() const { return kernel_type; }

    std::optional<JITOp> get_jit_op(int slot) const
    {
        switch (kernel_type)
        {
        case KernelType::SelectionKernelColumns:
            if (slot == 0) return JITOp(selection_columns_jit_0);
            if (slot == 1) return JITOp(selection_columns_jit_1);
            if (slot == 2) return JITOp(selection_columns_jit_2);
            if (slot == 3) return JITOp(selection_columns_jit_3);
            if (slot == 4) return JITOp(selection_columns_jit_4);
            if (slot == 5) return JITOp(selection_columns_jit_5);
            if (slot == 6) return JITOp(selection_columns_jit_6);
            if (slot == 7) return JITOp(selection_columns_jit_7);
            return std::nullopt;
        case KernelType::SelectionKernelLiteral:
            if (slot == 0) return JITOp(selection_literal_jit_0);
            if (slot == 1) return JITOp(selection_literal_jit_1);
            if (slot == 2) return JITOp(selection_literal_jit_2);
            if (slot == 3) return JITOp(selection_literal_jit_3);
            if (slot == 4) return JITOp(selection_literal_jit_4);
            if (slot == 5) return JITOp(selection_literal_jit_5);
            if (slot == 6) return JITOp(selection_literal_jit_6);
            if (slot == 7) return JITOp(selection_literal_jit_7);
            return std::nullopt;
        case KernelType::PerformOperationKernelColumns:
            if (slot == 0) return JITOp(perform_op_columns_jit_0);
            if (slot == 1) return JITOp(perform_op_columns_jit_1);
            if (slot == 2) return JITOp(perform_op_columns_jit_2);
            if (slot == 3) return JITOp(perform_op_columns_jit_3);
            if (slot == 4) return JITOp(perform_op_columns_jit_4);
            if (slot == 5) return JITOp(perform_op_columns_jit_5);
            if (slot == 6) return JITOp(perform_op_columns_jit_6);
            if (slot == 7) return JITOp(perform_op_columns_jit_7);
            return std::nullopt;
        case KernelType::PerformOperationKernelLiteralFirst:
            if (slot == 0) return JITOp(perform_op_literal_first_jit_0);
            if (slot == 1) return JITOp(perform_op_literal_first_jit_1);
            if (slot == 2) return JITOp(perform_op_literal_first_jit_2);
            if (slot == 3) return JITOp(perform_op_literal_first_jit_3);
            if (slot == 4) return JITOp(perform_op_literal_first_jit_4);
            if (slot == 5) return JITOp(perform_op_literal_first_jit_5);
            if (slot == 6) return JITOp(perform_op_literal_first_jit_6);
            if (slot == 7) return JITOp(perform_op_literal_first_jit_7);
            return std::nullopt;
        case KernelType::PerformOperationKernelLiteralSecond:
            if (slot == 0) return JITOp(perform_op_literal_second_jit_0);
            if (slot == 1) return JITOp(perform_op_literal_second_jit_1);
            if (slot == 2) return JITOp(perform_op_literal_second_jit_2);
            if (slot == 3) return JITOp(perform_op_literal_second_jit_3);
            if (slot == 4) return JITOp(perform_op_literal_second_jit_4);
            if (slot == 5) return JITOp(perform_op_literal_second_jit_5);
            if (slot == 6) return JITOp(perform_op_literal_second_jit_6);
            if (slot == 7) return JITOp(perform_op_literal_second_jit_7);
            return std::nullopt;
        case KernelType::FilterJoinKernel:
            if (slot == 0) return JITOp(filter_join_jit_0);
            if (slot == 1) return JITOp(filter_join_jit_1);
            if (slot == 2) return JITOp(filter_join_jit_2);
            if (slot == 3) return JITOp(filter_join_jit_3);
            if (slot == 4) return JITOp(filter_join_jit_4);
            if (slot == 5) return JITOp(filter_join_jit_5);
            if (slot == 6) return JITOp(filter_join_jit_6);
            if (slot == 7) return JITOp(filter_join_jit_7);
            return std::nullopt;
        case KernelType::FullJoinKernel:
            if (slot == 0) return JITOp(full_join_jit_0);
            if (slot == 1) return JITOp(full_join_jit_1);
            if (slot == 2) return JITOp(full_join_jit_2);
            if (slot == 3) return JITOp(full_join_jit_3);
            if (slot == 4) return JITOp(full_join_jit_4);
            if (slot == 5) return JITOp(full_join_jit_5);
            if (slot == 6) return JITOp(full_join_jit_6);
            if (slot == 7) return JITOp(full_join_jit_7);
            return std::nullopt;
        case KernelType::AggregateOperationKernel:
            if (slot == 0) return JITOp(aggregate_jit_0);
            if (slot == 1) return JITOp(aggregate_jit_1);
            if (slot == 2) return JITOp(aggregate_jit_2);
            if (slot == 3) return JITOp(aggregate_jit_3);
            if (slot == 4) return JITOp(aggregate_jit_4);
            if (slot == 5) return JITOp(aggregate_jit_5);
            if (slot == 6) return JITOp(aggregate_jit_6);
            if (slot == 7) return JITOp(aggregate_jit_7);
            return std::nullopt;
        default:
            return std::nullopt;
        }
    }

    void setup_jit_ctx(SYCLDBContext& ctx, int i) const {
        switch (kernel_type)
        {
        case KernelType::SelectionKernelColumns:
        {
            auto k = static_cast<SelectionKernelColumns *>(kernel_def.get());
            ctx.col_ptrs[i] = k->operand1;
            ctx.col_ptrs[i+8] = k->operand2;
            ctx.params[i*8+0] = (int)k->comparison;
            ctx.params[i*8+1] = (int)k->logic;
            break;
        }
        case KernelType::SelectionKernelLiteral:
        {
            auto k = static_cast<SelectionKernelLiteral *>(kernel_def.get());
            ctx.col_ptrs[i] = k->operand1;
            ctx.values[i] = k->value;
            ctx.params[i*8+0] = (int)k->comparison;
            ctx.params[i*8+1] = (int)k->logic;
            break;
        }
        case KernelType::PerformOperationKernelColumns:
        {
            auto k = static_cast<PerformOperationKernelColumns *>(kernel_def.get());
            ctx.res_ptrs[i] = k->result;
            ctx.col_ptrs[i] = k->col1;
            ctx.col_ptrs[i+8] = k->col2;
            ctx.params[i*8+2] = (int)k->op_enum;
            break;
        }
        case KernelType::PerformOperationKernelLiteralFirst:
        {
            auto k = static_cast<PerformOperationKernelLiteralFirst *>(kernel_def.get());
            ctx.res_ptrs[i] = k->result;
            ctx.values[i] = k->literal;
            ctx.col_ptrs[i] = k->col;
            ctx.params[i*8+2] = (int)k->op_enum;
            break;
        }
        case KernelType::PerformOperationKernelLiteralSecond:
        {
            auto k = static_cast<PerformOperationKernelLiteralSecond *>(kernel_def.get());
            ctx.res_ptrs[i] = k->result;
            ctx.col_ptrs[i] = k->col;
            ctx.values[i] = k->literal;
            ctx.params[i*8+2] = (int)k->op_enum;
            break;
        }
        case KernelType::FilterJoinKernel:
        {
            auto k = static_cast<FilterJoinKernel *>(kernel_def.get());
            ctx.col_ptrs[i] = k->probe_col;
            ctx.ht_ptrs[i] = k->build_ht;
            ctx.params[i*8+3] = k->build_min_value;
            ctx.params[i*8+4] = k->build_max_value;
            ctx.params[i*8+5] = k->ht_len;
            break;
        }
        case KernelType::FullJoinKernel:
        {
            auto k = static_cast<FullJoinKernel *>(kernel_def.get());
            ctx.col_ptrs[i] = k->probe_col;
            ctx.res_ptrs[i] = k->probe_val_out;
            ctx.ht_int_ptrs[i] = k->ht;
            ctx.params[i*8+3] = k->ht_min_value;
            ctx.params[i*8+4] = k->ht_max_value;
            ctx.params[i*8+5] = k->ht_len;
            break;
        }
        case KernelType::AggregateOperationKernel:
        {
            auto k = static_cast<AggregateOperationKernel *>(kernel_def.get());
            ctx.col_ptrs[i] = k->data;
            break;
        }
        default: break;
        }
    }

    uint64_t *get_agg_res_ptr() const {
        if (kernel_type == KernelType::AggregateOperationKernel) {
            return static_cast<AggregateOperationKernel *>(kernel_def.get())->agg_res;
        }
        return nullptr;
    }

    bool* get_input_flags() const {
        if (kernel_type == KernelType::SelectionKernelColumns) return const_cast<bool*>(static_cast<SelectionKernelColumns *>(kernel_def.get())->input_flags);
        if (kernel_type == KernelType::SelectionKernelLiteral) return const_cast<bool*>(static_cast<SelectionKernelLiteral *>(kernel_def.get())->input_flags);
        if (kernel_type == KernelType::FilterJoinKernel) return const_cast<bool*>(static_cast<FilterJoinKernel *>(kernel_def.get())->input_flags);
        if (kernel_type == KernelType::FullJoinKernel) return const_cast<bool*>(static_cast<FullJoinKernel *>(kernel_def.get())->input_flags);
        if (kernel_type == KernelType::AggregateOperationKernel) return const_cast<bool*>(static_cast<AggregateOperationKernel *>(kernel_def.get())->flags);
        if (kernel_type == KernelType::PerformOperationKernelColumns) return const_cast<bool*>(static_cast<PerformOperationKernelColumns *>(kernel_def.get())->flags);
        if (kernel_type == KernelType::PerformOperationKernelLiteralFirst) return const_cast<bool*>(static_cast<PerformOperationKernelLiteralFirst *>(kernel_def.get())->flags);
        if (kernel_type == KernelType::PerformOperationKernelLiteralSecond) return const_cast<bool*>(static_cast<PerformOperationKernelLiteralSecond *>(kernel_def.get())->flags);
        return nullptr;
    }

    bool* get_output_flags() const {
        if (kernel_type == KernelType::SelectionKernelColumns) return static_cast<SelectionKernelColumns *>(kernel_def.get())->output_flags;
        if (kernel_type == KernelType::SelectionKernelLiteral) return static_cast<SelectionKernelLiteral *>(kernel_def.get())->output_flags;
        if (kernel_type == KernelType::FilterJoinKernel) return static_cast<FilterJoinKernel *>(kernel_def.get())->output_flags;
        if (kernel_type == KernelType::FullJoinKernel) return static_cast<FullJoinKernel *>(kernel_def.get())->output_flags;
        return nullptr;
    }

    int get_col_len() const { return kernel_def->get_col_len(); }

    std::vector<sycl::event> execute(
        sycl::queue &queue,
        const std::vector<sycl::event> &dependencies
    ) const
    {
        switch (kernel_type)
        {
        case KernelType::EmptyKernel:
        {
            return dependencies;
        }
        case KernelType::LogicalKernel:
        {
            LogicalKernel *kernel = static_cast<LogicalKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::SelectionKernelColumns:
        {
            SelectionKernelColumns *kernel = static_cast<SelectionKernelColumns *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::SelectionKernelLiteral:
        {
            SelectionKernelLiteral *kernel = static_cast<SelectionKernelLiteral *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FillKernel:
        {
            FillKernel *kernel = static_cast<FillKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelColumns:
        {
            PerformOperationKernelColumns *kernel = static_cast<PerformOperationKernelColumns *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelLiteralFirst:
        {
            PerformOperationKernelLiteralFirst *kernel = static_cast<PerformOperationKernelLiteralFirst *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::PerformOperationKernelLiteralSecond:
        {
            PerformOperationKernelLiteralSecond *kernel = static_cast<PerformOperationKernelLiteralSecond *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::BuildKeysHTKernel:
        {
            BuildKeysHTKernel *kernel = static_cast<BuildKeysHTKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FilterJoinKernel:
        {
            FilterJoinKernel *kernel = static_cast<FilterJoinKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::BuildKeyValsHTKernel:
        {
            BuildKeyValsHTKernel *kernel = static_cast<BuildKeyValsHTKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::FullJoinKernel:
        {
            FullJoinKernel *kernel = static_cast<FullJoinKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::AggregateOperationKernel:
        {
            AggregateOperationKernel *kernel = static_cast<AggregateOperationKernel *>(kernel_def.get());
            uint64_t *agg_res_ptr = kernel->agg_res;
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        sycl::reduction(agg_res_ptr, sycl::plus<uint64_t>()),
                        * kernel
                    );
                }
            );
            return { e };
        }
        case KernelType::GroupByAggregateKernel:
        {
            GroupByAggregateKernel *kernel = static_cast<GroupByAggregateKernel *>(kernel_def.get());
            auto e = queue.submit(
                [&](sycl::handler &cgh)
                {
                    if (!dependencies.empty())
                        cgh.depends_on(dependencies);

                    cgh.parallel_for(
                        kernel->get_col_len(),
                        *kernel
                    );
                }
            );
            return { e };
        }
        default:
            throw std::invalid_argument("Unknown kernel type");
        }
    }
};

class KernelBundle
{
private:
    std::vector<KernelData> kernels;
    bool on_device;
    int device_index;
public:
    KernelBundle(bool on_device, int device_index)
        : on_device(on_device), device_index(device_index)
    {}

    bool is_on_device() const
    {
        return on_device;
    }

    int get_device_index() const
    {
        return device_index;
    }

    const std::vector<KernelData>& get_kernels() const { return kernels; }

    void add_kernel(KernelData kernel)
    {
        kernels.push_back(kernel);
    }

    std::vector<sycl::event> execute(
        sycl::queue &cpu_queue,
        std::vector<sycl::queue> &device_queues,
        const std::vector<sycl::event> &cpu_dependencies,
        const std::vector<std::vector<sycl::event>> &device_dependencies
    ) const
    {
        std::vector<sycl::event> deps = on_device ? device_dependencies[device_index] : cpu_dependencies;
        sycl::queue &q = on_device ? device_queues[device_index] : cpu_queue;

        if (on_device) {
            std::vector<JITOp> jit_ops;
            SYCLDBContext ctx = {0};
            uint64_t *agg_res_ptr = nullptr;
            bool *initial_flags = nullptr;
            bool *final_flags = nullptr;
            int col_len = 0;

            static bool disable_fusion = (std::getenv("SYCLDB_DISABLE_FUSION") != nullptr);

            int split_index = 0;
            if (!disable_fusion) {
                for (int i = 0; i < kernels.size(); ++i) {
                    auto op = kernels[i].get_jit_op(i);
                    if (op && i < 8) {
                        jit_ops.push_back(*op);
                        kernels[i].setup_jit_ctx(ctx, i);
                        if (agg_res_ptr == nullptr) agg_res_ptr = kernels[i].get_agg_res_ptr();
                        if (initial_flags == nullptr) initial_flags = kernels[i].get_input_flags();
                        final_flags = kernels[i].get_output_flags();
                        if (col_len == 0) col_len = kernels[i].get_col_len();
                        split_index = i + 1;
                    } else {
                        // Stop at first non-JIT capable kernel or if we exceed 8 slots
                        break;
                    }
                }
            }

            if (split_index > 1) {
                auto jit_start = std::chrono::high_resolution_clock::now();
                acpp_jit::dynamic_function_config cfg;
                cfg.define_as_call_sequence(&execute_sycldb_ops_agg, jit_ops);
                
                auto jit_kernel = cfg.apply([=](sycl::item<1> idx, auto& acc_red) {
                    bool pass = initial_flags ? initial_flags[idx] : true;
                    uint64_t acc = 0;
                    execute_sycldb_ops_agg(idx, ctx, pass, acc);
                    acc_red += acc;
                    if (final_flags) final_flags[idx] = pass;
                });
                auto jit_end = std::chrono::high_resolution_clock::now();
                double jit_time = std::chrono::duration<double, std::milli>(jit_end - jit_start).count();

                auto kernel_start = std::chrono::high_resolution_clock::now();
                auto e = q.submit([&](sycl::handler &cgh) {
                    if (!deps.empty()) cgh.depends_on(deps);
                    if (agg_res_ptr) {
                        cgh.parallel_for(sycl::range<1>{(size_t)col_len}, sycl::reduction(agg_res_ptr, sycl::plus<uint64_t>()), 
                            jit_kernel);
                    } else {
                        // Re-bind without reduction if not needed
                        cgh.parallel_for(sycl::range<1>{(size_t)col_len}, 
                            cfg.apply([=](sycl::item<1> idx) {
                                bool pass = initial_flags ? initial_flags[idx] : true;
                                uint64_t acc = 0;
                                execute_sycldb_ops_agg(idx, ctx, pass, acc);
                                if (final_flags) final_flags[idx] = pass;
                            }));
                    }
                });
                e.wait();
                auto kernel_end = std::chrono::high_resolution_clock::now();
                double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
                g_total_jit_ms += jit_time;
                g_total_kernel_ms += kernel_time;
                
                std::cout << "[JIT-FUSION] Partial fusion (size " << split_index << ") JIT time: " << jit_time << " ms, Kernel time: " << kernel_time << " ms" << std::endl;
                deps = {e};
            } else {
                split_index = 0; // Did not fuse anything (or only 1, which we don't JIT yet)
            }

            if (split_index < kernels.size()) {
                auto kernel_fallback_start = std::chrono::high_resolution_clock::now();
                for (int i = split_index; i < kernels.size(); ++i) {
                    deps = kernels[i].execute(q, deps);
                }
                // Wait for fallback kernels
                for (auto &e : deps) e.wait();
                auto kernel_fallback_end = std::chrono::high_resolution_clock::now();
                g_total_kernel_ms += std::chrono::duration<double, std::milli>(kernel_fallback_end - kernel_fallback_start).count();
            }
            return deps;
        }

        // Host execution path
        auto kernel_fallback_start = std::chrono::high_resolution_clock::now();
        for (const KernelData &kernel : kernels)
        {
            deps = kernel.execute(q, deps);
        }
        for (auto &e : deps) e.wait();
        auto kernel_fallback_end = std::chrono::high_resolution_clock::now();
        g_total_kernel_ms += std::chrono::duration<double, std::milli>(kernel_fallback_end - kernel_fallback_start).count();

        return deps;
    }
};
