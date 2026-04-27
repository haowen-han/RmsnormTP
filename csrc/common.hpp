#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rmsnorm_tp {

// Constants
constexpr int MAX_TP_SIZE = 8;
constexpr int HANDLE_SIZE = 128;  // Enough for cudaIpcMemHandle_t

// Ipc handle storage (plain old data for Python serialization)
struct IpcHandle {
    char data[HANDLE_SIZE];
    size_t alloc_size;
};

}  // namespace rmsnorm_tp
