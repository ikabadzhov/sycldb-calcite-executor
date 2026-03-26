#!/bin/bash

set -euo pipefail

ensure_benchmark_env() {
    if [[ -z "${ONEAPI_ROOT:-}" && -z "${CMPLR_ROOT:-}" && -z "${DPCPP_ROOT:-}" ]]; then
        source /home/eugenio/develop-env-vars.sh >/dev/null 2>&1
    fi

    local extra_ld_library_path
    extra_ld_library_path="/opt/intel/oneapi/tcm/1.3/lib:/opt/intel/oneapi/umf/0.10/lib:/opt/intel/oneapi/tbb/2022.1/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.11/lib:/opt/intel/oneapi/mpi/2021.15/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.15/lib:/opt/intel/oneapi/mkl/2025.1/lib:/opt/intel/oneapi/ippcp/2025.1/lib/:/opt/intel/oneapi/ipp/2022.1/lib:/opt/intel/oneapi/dnnl/2025.1/lib:/opt/intel/oneapi/debugger/2025.1/opt/debugger/lib:/opt/intel/oneapi/dal/2025.4/lib:/opt/intel/oneapi/compiler/2025.1/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.1/lib:/opt/intel/oneapi/ccl/2021.15/lib/"

    if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        export LD_LIBRARY_PATH="${extra_ld_library_path}:${LD_LIBRARY_PATH}"
    else
        export LD_LIBRARY_PATH="${extra_ld_library_path}"
    fi
}
