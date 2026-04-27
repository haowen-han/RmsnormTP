#include "common.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK AT_CUDA_CHECK

#include <cstring>
#include <vector>
#include <string>

namespace rmsnorm_tp {

// Manage IPC shared memory for inter-GPU communication within a node
class CommBuffer {
public:
    CommBuffer(int rank, int tp_size, int64_t num_max_rows)
        : rank_(rank), tp_size_(tp_size), num_max_rows_(num_max_rows),
          initialized_(false) {}

    ~CommBuffer() { destroy(); }

    void allocate() {
        // Each GPU's buffer layout:
        //   float partial_sums[tp_size][num_max_rows]  -- mailbox slots, one per source rank
        //   int   generation[tp_size][num_max_rows]    -- generation counters
        size_t sums_bytes = (size_t)tp_size_ * num_max_rows_ * sizeof(float);
        size_t gen_bytes = (size_t)tp_size_ * num_max_rows_ * sizeof(int);
        alloc_size_ = sums_bytes + gen_bytes;

        CUDA_CHECK(cudaMalloc(&local_buffer_, alloc_size_));
        CUDA_CHECK(cudaMemset(local_buffer_, 0, alloc_size_));

        local_sums_ = reinterpret_cast<float*>(local_buffer_);
        local_gen_ = reinterpret_cast<int*>(static_cast<uint8_t*>(local_buffer_) + sums_bytes);

        // Initialize pointer arrays
        for (int i = 0; i < MAX_TP_SIZE; i++) {
            sums_ptrs_[i] = nullptr;
            gen_ptrs_[i] = nullptr;
        }
        sums_ptrs_[rank_] = local_sums_;
        gen_ptrs_[rank_] = local_gen_;

        // Allocate GPU-side pointer arrays
        CUDA_CHECK(cudaMalloc(&sums_ptrs_gpu_, MAX_TP_SIZE * sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&gen_ptrs_gpu_, MAX_TP_SIZE * sizeof(int*)));
        CUDA_CHECK(cudaMemcpy(sums_ptrs_gpu_, sums_ptrs_, MAX_TP_SIZE * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gen_ptrs_gpu_, gen_ptrs_, MAX_TP_SIZE * sizeof(int*), cudaMemcpyHostToDevice));
    }

    IpcHandle get_ipc_handle() {
        IpcHandle handle;
        handle.alloc_size = alloc_size_;
        CUDA_CHECK(cudaIpcGetMemHandle(
            reinterpret_cast<cudaIpcMemHandle_t*>(handle.data), local_buffer_));
        return handle;
    }

    void open_remote_handle(int remote_rank, const IpcHandle& handle) {
        void* remote_ptr;
        CUDA_CHECK(cudaIpcOpenMemHandle(&remote_ptr,
            *reinterpret_cast<const cudaIpcMemHandle_t*>(handle.data),
            cudaIpcMemLazyEnablePeerAccess));

        size_t sums_bytes = (size_t)tp_size_ * num_max_rows_ * sizeof(float);
        sums_ptrs_[remote_rank] = reinterpret_cast<float*>(remote_ptr);
        gen_ptrs_[remote_rank] = reinterpret_cast<int*>(
            static_cast<uint8_t*>(remote_ptr) + sums_bytes);

        // Update GPU-side arrays
        CUDA_CHECK(cudaMemcpy(sums_ptrs_gpu_, sums_ptrs_, MAX_TP_SIZE * sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gen_ptrs_gpu_, gen_ptrs_, MAX_TP_SIZE * sizeof(int*), cudaMemcpyHostToDevice));
    }

    void finalize_setup() {
        CUDA_CHECK(cudaDeviceSynchronize());
        initialized_ = true;
    }

    void destroy() {
        if (!local_buffer_) return;
        // Close remote handles
        for (int i = 0; i < tp_size_; i++) {
            if (i != rank_ && sums_ptrs_[i]) {
                cudaIpcCloseMemHandle(sums_ptrs_[i]);
                sums_ptrs_[i] = nullptr;
                gen_ptrs_[i] = nullptr;
            }
        }
        CUDA_CHECK(cudaFree(local_buffer_));
        local_buffer_ = nullptr;
        if (sums_ptrs_gpu_) { CUDA_CHECK(cudaFree(sums_ptrs_gpu_)); sums_ptrs_gpu_ = nullptr; }
        if (gen_ptrs_gpu_) { CUDA_CHECK(cudaFree(gen_ptrs_gpu_)); gen_ptrs_gpu_ = nullptr; }
        initialized_ = false;
    }

    // Getters for kernel launch
    float** sums_ptrs_gpu() const { return sums_ptrs_gpu_; }
    int** gen_ptrs_gpu() const { return gen_ptrs_gpu_; }
    int rank() const { return rank_; }
    int tp_size() const { return tp_size_; }
    int64_t num_max_rows() const { return num_max_rows_; }
    bool initialized() const { return initialized_; }

private:
    int rank_;
    int tp_size_;
    int64_t num_max_rows_;
    bool initialized_;
    size_t alloc_size_ = 0;

    void* local_buffer_ = nullptr;
    float* local_sums_ = nullptr;
    int* local_gen_ = nullptr;

    float* sums_ptrs_[MAX_TP_SIZE];
    int* gen_ptrs_[MAX_TP_SIZE];
    float** sums_ptrs_gpu_ = nullptr;
    int** gen_ptrs_gpu_ = nullptr;
};

}  // namespace rmsnorm_tp

// Kernel launcher declaration (in namespace, matches .cu definition)
namespace rmsnorm_tp {

void launch_fused_rmsnorm(
    float* output,
    const float* input,
    const float* weight,
    float** sums_ptrs_gpu,
    int** gen_ptrs_gpu,
    int rank,
    int tp_size,
    int local_hidden_dim,
    int total_hidden_dim,
    int num_rows,
    float eps,
    int generation,
    cudaStream_t stream);

}  // namespace rmsnorm_tp

namespace py = pybind11;

PYBIND11_MODULE(rmsnorm_tp_cpp, m) {
    py::class_<rmsnorm_tp::IpcHandle>(m, "IpcHandle")
        .def(py::init<>())
        .def_readonly("alloc_size", &rmsnorm_tp::IpcHandle::alloc_size)
        .def("to_bytes", [](const rmsnorm_tp::IpcHandle& h) -> py::bytes {
            return py::bytes(h.data, rmsnorm_tp::HANDLE_SIZE);
        })
        .def("from_bytes", [](rmsnorm_tp::IpcHandle& h, py::bytes b) {
            std::string s = b;
            if (s.size() != rmsnorm_tp::HANDLE_SIZE)
                throw std::runtime_error("Invalid handle size");
            std::memcpy(h.data, s.data(), rmsnorm_tp::HANDLE_SIZE);
        });

    py::class_<rmsnorm_tp::CommBuffer>(m, "CommBuffer")
        .def(py::init<int, int, int64_t>())
        .def("allocate", &rmsnorm_tp::CommBuffer::allocate)
        .def("get_ipc_handle", &rmsnorm_tp::CommBuffer::get_ipc_handle)
        .def("open_remote_handle", &rmsnorm_tp::CommBuffer::open_remote_handle)
        .def("finalize_setup", &rmsnorm_tp::CommBuffer::finalize_setup)
        .def("destroy", &rmsnorm_tp::CommBuffer::destroy)
        .def("sums_ptrs_gpu", [](rmsnorm_tp::CommBuffer& buf) -> int64_t {
            return reinterpret_cast<int64_t>(buf.sums_ptrs_gpu());
        })
        .def("gen_ptrs_gpu", [](rmsnorm_tp::CommBuffer& buf) -> int64_t {
            return reinterpret_cast<int64_t>(buf.gen_ptrs_gpu());
        })
        .def("rank", &rmsnorm_tp::CommBuffer::rank)
        .def("tp_size", &rmsnorm_tp::CommBuffer::tp_size)
        .def("num_max_rows", &rmsnorm_tp::CommBuffer::num_max_rows)
        .def("initialized", &rmsnorm_tp::CommBuffer::initialized);

    m.def("launch_fused_rmsnorm", [](int64_t output_ptr, int64_t input_ptr, int64_t weight_ptr,
                                      int64_t sums_ptrs_gpu, int64_t gen_ptrs_gpu,
                                      int rank, int tp_size, int local_hidden_dim,
                                      int total_hidden_dim, int num_rows, float eps,
                                      int generation, int64_t stream_ptr) {
        rmsnorm_tp::launch_fused_rmsnorm(
            reinterpret_cast<float*>(output_ptr),
            reinterpret_cast<const float*>(input_ptr),
            reinterpret_cast<const float*>(weight_ptr),
            reinterpret_cast<float**>(sums_ptrs_gpu),
            reinterpret_cast<int**>(gen_ptrs_gpu),
            rank, tp_size, local_hidden_dim, total_hidden_dim,
            num_rows, eps, generation,
            reinterpret_cast<cudaStream_t>(stream_ptr));
    });
}
