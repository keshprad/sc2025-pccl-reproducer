/*
 * GPU-triggered communication queue implementation using HIP
 * 
 * This implementation provides a shared queue system between GPU and host:
 * - GPU enqueues requests and dequeues completions
 * - Host dequeues requests and enqueues completions
 * 
 * Key HIP features used:
 * - hipHostMalloc with hipHostMallocMapped for zero-copy memory
 * - __threadfence_system() for system-wide memory ordering
 * - hipLaunchKernelGGL for kernel launches
 */

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <new>
#include <hip/hip_runtime.h>
#include <iostream>

// Power-of-two ring sizes make index wrap cheap
#define REQ_QUEUE_SIZE  1024u
#define CMP_QUEUE_SIZE  1024u

// HIP error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

enum RequestType {
    SEND = 0,
    RECV = 1
};

enum RequestStatus {
    PENDING = 0,
    COMPLETE = 1
};

struct alignas(64) RequestQueueEntry {
    int request_id;
    int origin_rank; // ID of the GPU rank that initiated this request
    int peer_rank;
    void *buffer;
    size_t size;
    RequestType type;
    RequestStatus status;
    int dirty; // dirty bit; 0 = free, 1 = ready

    RequestQueueEntry() : request_id(0), origin_rank(-1), peer_rank(-1), buffer(nullptr), size(0), type(SEND), status(PENDING), dirty(0) {}
    
    RequestQueueEntry(int request_id, int origin_rank, int peer_rank, void *buffer, size_t size, RequestType type) :
        request_id(request_id), origin_rank(origin_rank), peer_rank(peer_rank), buffer(buffer), size(size), type(type), status(PENDING), dirty(1) {}
};

struct alignas(64) CompletionQueueEntry {
    int request_id;
    int origin_rank; // ID of the GPU rank that should receive this completion
    int bytes_transferred;
    // TODO: replace with status enum
    int status;
    int dirty; // dirty bit; 0 = free, 1 = ready

    CompletionQueueEntry() : request_id(0), origin_rank(-1), bytes_transferred(0), status(0), dirty(0) {}
    
    CompletionQueueEntry(int request_id, int origin_rank, int bytes_transferred, int status) :
        request_id(request_id), origin_rank(-1), bytes_transferred(bytes_transferred), status(status), dirty(1) {}
};

class SharedQueue {
private:
    // Host-pinned, GPU-mapped memory for queues
    RequestQueueEntry *h_request_queue, *d_request_queue;
    CompletionQueueEntry *h_completion_queue, *d_completion_queue;
    // Queue management pointers (also host-pinned, GPU-mapped)
    int *h_request_head, *d_request_head;
    int *h_request_tail, *d_request_tail;
    int *h_completion_head, *d_completion_head;
    int *h_completion_tail, *d_completion_tail;
    
    size_t req_queue_size;
    size_t cmp_queue_size;

public:
    SharedQueue() : req_queue_size(REQ_QUEUE_SIZE), cmp_queue_size(CMP_QUEUE_SIZE) {
        initialize();
    }
    
    ~SharedQueue() {
        cleanup();
    }
    
    void initialize();
    void cleanup();
    
    // Device-side functions (called from GPU kernels)
    __device__ bool enqueue_request(int request_id, int origin_rank, int peer_rank, void *buffer, size_t size, RequestType type);
    __device__ bool dequeue_completion(CompletionQueueEntry &entry, int gpu_rank);
    
    // Host-side functions
    bool dequeue_request(RequestQueueEntry &entry);
    bool enqueue_completion(int request_id, int origin_rank, int bytes_transferred, int status);
    
    // Utility functions for circular queue management
    bool is_request_queue_empty();
    bool is_completion_queue_empty();
    size_t get_request_queue_size();
    size_t get_completion_queue_size();
    bool is_request_queue_full();
    bool is_completion_queue_full();
    size_t get_request_queue_free_space();
    size_t get_completion_queue_free_space();
};

// SharedQueue implementation
void SharedQueue::initialize() {
    // Allocate host-pinned memory for request queue
    HIP_CHECK(hipHostMalloc((void**)&h_request_queue, 
                           req_queue_size * sizeof(RequestQueueEntry), 
                           hipHostMallocMapped));
    // Allocate host-pinned memory for completion queue
    HIP_CHECK(hipHostMalloc((void**)&h_completion_queue, 
                            cmp_queue_size * sizeof(CompletionQueueEntry), 
                            hipHostMallocMapped));
    // Get GPU-mapped device pointers
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_request_queue, h_request_queue, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_completion_queue, h_completion_queue, 0));
    
    // Allocate host-pinned memory for queue pointers
    HIP_CHECK(hipHostMalloc((void**)&h_request_head, sizeof(int), hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc((void**)&h_request_tail, sizeof(int), hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc((void**)&h_completion_head, sizeof(int), hipHostMallocMapped));
    HIP_CHECK(hipHostMalloc((void**)&h_completion_tail, sizeof(int), hipHostMallocMapped));
    // Get GPU-mapped device pointers
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_request_head, h_request_head, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_request_tail, h_request_tail, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_completion_head, h_completion_head, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_completion_tail, h_completion_tail, 0));
    
    // Initialize queue pointers
    *h_request_head = 0;
    *h_request_tail = 0;
    *h_completion_head = 0;
    *h_completion_tail = 0;
    
    // Initialize queue entries
    for (size_t i = 0; i < req_queue_size; ++i) {
        h_request_queue[i] = RequestQueueEntry();
    }
    for (size_t i = 0; i < cmp_queue_size; ++i) {
        h_completion_queue[i] = CompletionQueueEntry();
    }
}

void SharedQueue::cleanup() {
    if (h_request_queue) hipHostFree(h_request_queue);
    if (h_completion_queue) hipHostFree(h_completion_queue);
    if (h_request_head) hipHostFree(h_request_head);
    if (h_request_tail) hipHostFree(h_request_tail);
    if (h_completion_head) hipHostFree(h_completion_head);
    if (h_completion_tail) hipHostFree(h_completion_tail);
}

// Device-side function: GPU enqueues requests
__device__ bool SharedQueue::enqueue_request(int request_id, int origin_rank, int peer_rank, void *buffer, size_t size, RequestType type) {
    int current_tail, next_tail;
    int slot;
    
    // use CAS to acquire unique, open slot in request queue
    do {
        current_tail = *d_request_tail;
        slot = current_tail & (req_queue_size - 1);
        
        // TODO: handle queue full properly rather than failure
        // Check if queue is full
        if (current_tail - *d_request_head >= req_queue_size) {
            return false;
        }
        
        // Check if slot is free
        if (d_request_queue[slot].dirty != 0) {
            return false;
        }

        // advance tail
        next_tail = current_tail + 1;
        
    } while (atomicCAS(d_request_tail, current_tail, next_tail) != current_tail);
    
    // Construct the entry in-place using placement new
    new (&d_request_queue[slot]) RequestQueueEntry(request_id, origin_rank, peer_rank, buffer, size, type);
    
    __threadfence_system(); // Ensure all writes are visible to host
    d_request_queue[slot].dirty = 1; // Mark as ready
    
    return true;
}

// Device-side function: GPU dequeues completion for the current rank
__device__ bool SharedQueue::dequeue_completion(CompletionQueueEntry &entry, int rank) {
    int current_head, next_head, slot;
    
    // Use CAS to atomically check and advance head pointer in completion queue
    do {
        current_head = *d_completion_head;
        slot = current_head & (cmp_queue_size - 1);
        
        // Check if queue is empty
        if (current_head >= *d_completion_tail) {
            return false;
        }
        
        // Check if entry is not ready
        if (d_completion_queue[slot].dirty == 0) {
            return false;
        }
        
        // Check if the head entry doesn't belong to this GPU rank
        if (d_completion_queue[slot].origin_rank != rank) {
            return false;
        }
        
        // advance head
        next_head = current_head + 1;

    } while (atomicCAS(d_completion_head, current_head, next_head) != current_head);
    
    // Successfully claimed the head entry
    entry = d_completion_queue[slot];
    __threadfence_system(); // Ensure read is complete
    
    d_completion_queue[slot].dirty = 0; // Mark as consumed
    
    return true;
}

// Host-side function: Host dequeues requests
bool SharedQueue::dequeue_request(RequestQueueEntry &entry) {
    int current_head, next_head, slot;
    
    // Use CAS to atomically check and advance head pointer in request queue
    do {
        current_head = *h_request_head;
        slot = current_head & (req_queue_size - 1);
        
        // Check if queue is empty
        if (current_head >= *h_request_tail) {
            return false;
        }
        
        // Check if entry is not ready
        if (h_request_queue[slot].dirty == 0) {
            return false;
        }

        // advance head
        next_head = current_head + 1;

    } while (__atomic_compare_exchange_n(h_request_head, &current_head, next_head, 
                                        false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) == false);
    
    // Successfully claimed the head entry
    entry = h_request_queue[slot];
    
    h_request_queue[slot].dirty = 0; // Mark as consumed
    
    return true;
}

// Host-side function: Host enqueues completions
bool SharedQueue::enqueue_completion(int request_id, int origin_rank, int bytes_transferred, int status) {
    int current_tail, next_tail;
    int slot;
    
    // Use CAS to acquire unique, open slot in completion queue
    do {
        current_tail = *h_completion_tail;
        slot = current_tail & (cmp_queue_size - 1);
        
        // Check if queue is full
        if (current_tail - *h_completion_head >= cmp_queue_size) {
            return false;
        }
        
        // Check if slot is free
        if (h_completion_queue[slot].dirty != 0) {
            return false;
        }

        // advance tail
        next_tail = current_tail + 1;
        
    } while (__atomic_compare_exchange_n(h_completion_tail, &current_tail, next_tail, 
                                        false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE) == false);
    
    // Construct the entry in-place using placement new
    new (&h_completion_queue[slot]) CompletionQueueEntry(request_id, origin_rank, bytes_transferred, status);
    
    h_completion_queue[slot].dirty = 1; // Mark as ready
    
    return true;
}

/*
 * Thread-safe utility functions
 * 
 * These functions read shared data that's being modified concurrently by:
 * - GPU threads (modifying head/tail pointers atomically)
 * - Host threads (modifying head/tail pointers)
 * 
 * Race condition risks without proper synchronization:
 * 1. Torn reads: Reading head and tail at different times
 * 2. Inconsistent snapshots: head/tail changing between reads
 * 3. Compiler optimizations reordering reads
 * 
 * Solutions implemented:
 * 1. Atomic loads with acquire semantics (__atomic_load_n)
 * 2. Consistent snapshots by reading head and tail atomically
 * 3. Memory ordering prevents compiler reordering
 */

// Utility functions - Thread-safe versions with atomic reads
bool SharedQueue::is_request_queue_empty() {
    // Atomic read of both head and tail to avoid torn reads
    int head = __atomic_load_n(h_request_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_request_tail, __ATOMIC_ACQUIRE);
    return (head == tail);
}

bool SharedQueue::is_completion_queue_empty() {
    // Atomic read of both head and tail to avoid torn reads
    int head = __atomic_load_n(h_completion_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_completion_tail, __ATOMIC_ACQUIRE);
    return (head == tail);
}

size_t SharedQueue::get_request_queue_size() {
    // Atomic snapshot of head and tail
    int head = __atomic_load_n(h_request_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_request_tail, __ATOMIC_ACQUIRE);
    int size = tail - head;
    return (size >= 0) ? size : 0;  // Ensure non-negative
}

size_t SharedQueue::get_completion_queue_size() {
    // Atomic snapshot of head and tail
    int head = __atomic_load_n(h_completion_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_completion_tail, __ATOMIC_ACQUIRE);
    int size = tail - head;
    return (size >= 0) ? size : 0;  // Ensure non-negative
}
bool SharedQueue::is_request_queue_full() {
    // Use atomic reads for consistent snapshot
    int head = __atomic_load_n(h_request_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_request_tail, __ATOMIC_ACQUIRE);
    return ((tail - head) >= REQ_QUEUE_SIZE);
}

bool SharedQueue::is_completion_queue_full() {
    // Use atomic reads for consistent snapshot
    int head = __atomic_load_n(h_completion_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_completion_tail, __ATOMIC_ACQUIRE);
    return ((tail - head) >= CMP_QUEUE_SIZE);
}

size_t SharedQueue::get_request_queue_free_space() {
    // Use atomic reads for consistent snapshot
    int head = __atomic_load_n(h_request_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_request_tail, __ATOMIC_ACQUIRE);
    int used = tail - head;
    return (used < REQ_QUEUE_SIZE) ? (REQ_QUEUE_SIZE - used) : 0;
}

size_t SharedQueue::get_completion_queue_free_space() {
    // Use atomic reads for consistent snapshot
    int head = __atomic_load_n(h_completion_head, __ATOMIC_ACQUIRE);
    int tail = __atomic_load_n(h_completion_tail, __ATOMIC_ACQUIRE);
    int used = tail - head;
    return (used < CMP_QUEUE_SIZE) ? (CMP_QUEUE_SIZE - used) : 0;
}

// Example GPU kernel that enqueues requests
__global__ void gpu_communication_kernel(SharedQueue *queue, int rank, void *send_buffer, size_t buffer_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread on this GPU enqueues a request with the GPU's rank
    if (tid < 10) { // Limit to first 10 threads for example
        bool success = queue->enqueue_request(
            rank * 1000 + tid,  // request_id (unique per GPU rank)
            rank                // originating GPU rank
            tid % 4,                   // peer_rank (example: 4 peers)
            send_buffer,               // buffer
            buffer_size,               // size
            SEND,                      // type
        );
        
        if (success) {
            printf("GPU rank %d thread %d: Successfully enqueued send request\n", rank, tid);
        }
        
        // Example: Check for completions belonging to this GPU rank
        CompletionQueueEntry completion;
        if (queue->dequeue_completion(completion, rank)) {
            printf("GPU rank %d thread %d: Received completion for request %d (status: %d, bytes: %d)\n", 
                   rank, tid, completion.request_id, completion.status, completion.bytes_transferred);
        }
    }
}

// Example host function that processes requests
void host_process_requests(SharedQueue *queue) {
    RequestQueueEntry request;
    
    // Process all pending requests
    while (queue->dequeue_request(request)) {
        std::cout << "Host: Processing request " << request.request_id 
                  << " from GPU rank " << request.origin_rank
                  << " peer " << request.peer_rank 
                  << " size " << request.size 
                  << " type " << (request.type == SEND ? "SEND" : "RECV") 
                  << std::endl;
        
        // Simulate processing and enqueue completion back to the originating GPU rank
        int bytes_transferred = request.size;
        int status = 0; // Success
        
        queue->enqueue_completion(request.request_id, request.origin_rank, bytes_transferred, status);
        std::cout << "Host: Enqueued completion for request " << request.request_id 
                  << " back to GPU rank " << request.origin_rank << std::endl;
    }
}

// Example main function
int main() {
    std::cout << "Starting HIP GPU-triggered communication queue test..." << std::endl;
    
    SharedQueue queue;
    
    // Allocate some dummy buffer
    void *send_buffer;
    size_t buffer_size = 1024;
    HIP_CHECK(hipMalloc(&send_buffer, buffer_size));
    
    // Simulate GPU rank 0 (in a real multi-GPU setup, this would be determined by the actual GPU)
    int my_gpu_rank = 0;
    
    std::cout << "Launching GPU kernel for GPU rank " << my_gpu_rank << "..." << std::endl;
    
    // Launch GPU kernel with GPU rank
    dim3 block(32);
    dim3 grid(1);
    gpu_communication_kernel<<<grid, block>>>(&queue, send_buffer, buffer_size, my_gpu_rank);
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "GPU kernel completed. Processing requests on host..." << std::endl;
    
    // Process requests on host
    host_process_requests(&queue);
    
    std::cout << "Queue Statistics:" << std::endl;
    std::cout << "  Request queue size: " << queue.get_request_queue_size() << std::endl;
    std::cout << "  Request queue free space: " << queue.get_request_queue_free_space() << std::endl;
    std::cout << "  Request queue empty: " << (queue.is_request_queue_empty() ? "Yes" : "No") << std::endl;
    std::cout << "  Request queue full: " << (queue.is_request_queue_full() ? "Yes" : "No") << std::endl;
    
    std::cout << "  Completion queue size: " << queue.get_completion_queue_size() << std::endl;
    std::cout << "  Completion queue free space: " << queue.get_completion_queue_free_space() << std::endl;
    std::cout << "  Completion queue empty: " << (queue.is_completion_queue_empty() ? "Yes" : "No") << std::endl;
    std::cout << "  Completion queue full: " << (queue.is_completion_queue_full() ? "Yes" : "No") << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(send_buffer));
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}

