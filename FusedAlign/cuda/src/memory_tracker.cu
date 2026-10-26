#include "cuda_helper.cuh"

#ifdef DEBUG

namespace {

struct CudaMemoryTrackerState {
    std::unordered_map<void*, size_t> live_allocations;
    size_t current_bytes = 0;
    size_t peak_bytes = 0;
    size_t allocation_calls = 0;
    size_t free_calls = 0;
    size_t total_allocated_bytes = 0;
    size_t total_freed_bytes = 0;
};

CudaMemoryTrackerState& tracker_state() {
    static CudaMemoryTrackerState state;
    return state;
}

}  // namespace

void cuda_memtracker_reset() {
    CudaMemoryTrackerState& state = tracker_state();
    state.live_allocations.clear();
    state.current_bytes = 0;
    state.peak_bytes = 0;
    state.allocation_calls = 0;
    state.free_calls = 0;
    state.total_allocated_bytes = 0;
    state.total_freed_bytes = 0;
}

void cuda_memtracker_report_peak(const char* label) {
    const CudaMemoryTrackerState& state = tracker_state();
    const char* title = (label != nullptr) ? label : "entire run";
    const double mib = static_cast<double>(state.peak_bytes) / (1024.0 * 1024.0);
    std::cout << "gpu allocation tracker (" << title << "):\n"
              << "  peak tracked bytes: " << state.peak_bytes << " (" << mib << " MiB)\n"
              << "  current tracked bytes: " << state.current_bytes << '\n'
              << "  allocation calls: " << state.allocation_calls << '\n'
              << "  free calls: " << state.free_calls << '\n'
              << "  total allocated bytes: " << state.total_allocated_bytes << '\n'
              << "  total freed bytes: " << state.total_freed_bytes << '\n'
              << "  live allocations: " << state.live_allocations.size()
              << std::endl;
}

cudaError_t cuda_malloc_tracked(void** ptr, size_t bytes, const char* file, int line) {
    cudaError_t status = cudaMalloc(ptr, bytes);
    if (status != cudaSuccess || ptr == nullptr || *ptr == nullptr) {
        return status;
    }

    CudaMemoryTrackerState& state = tracker_state();
    const auto it = state.live_allocations.find(*ptr);
    if (it != state.live_allocations.end()) {
        state.current_bytes -= it->second;
        state.total_freed_bytes += it->second;
        state.live_allocations.erase(it);
        DEBUG_ONLY(std::cerr << "gpu tracker warning: pointer reused without tracked free at "
                             << file << ":" << line << std::endl);
    }

    state.live_allocations[*ptr] = bytes;
    state.current_bytes += bytes;
    state.total_allocated_bytes += bytes;
    state.allocation_calls += 1;
    state.peak_bytes = std::max(state.peak_bytes, state.current_bytes);
    return status;
}

cudaError_t cuda_free_tracked(void* ptr, const char* file, int line) {
    cudaError_t status = cudaFree(ptr);
    if (status != cudaSuccess || ptr == nullptr) {
        return status;
    }

    CudaMemoryTrackerState& state = tracker_state();
    const auto it = state.live_allocations.find(ptr);
    if (it != state.live_allocations.end()) {
        state.current_bytes -= it->second;
        state.total_freed_bytes += it->second;
        state.live_allocations.erase(it);
    } else {
        DEBUG_ONLY(std::cerr << "gpu tracker warning: untracked free at "
                             << file << ":" << line << std::endl);
    }
    state.free_calls += 1;
    return status;
}

#endif
