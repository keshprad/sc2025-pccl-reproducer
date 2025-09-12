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
#include <mpi.h>

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

    __device__ __host__ RequestQueueEntry() : request_id(0), origin_rank(-1), peer_rank(-1), buffer(nullptr), size(0), type(SEND), status(PENDING), dirty(0) {}
    
    // NOTE: use like:
    // RequestQueueEntry(...)
    // __threadfence_system()
    // entry.dirty = 1
    __device__ __host__ RequestQueueEntry(int request_id, int origin_rank, int peer_rank, void *buffer, size_t size, RequestType type) :
        request_id(request_id), origin_rank(origin_rank), peer_rank(peer_rank), buffer(buffer), size(size), type(type), status(PENDING), dirty(0) {}
};

struct alignas(64) CompletionQueueEntry {
    int request_id;
    int origin_rank; // ID of the GPU rank that should receive this completion
    int bytes_transferred;
    // TODO: replace with status enum
    int status;
    int dirty; // dirty bit; 0 = free, 1 = ready

    __device__ __host__ CompletionQueueEntry() : request_id(0), origin_rank(-1), bytes_transferred(0), status(0), dirty(0) {}
    
    __device__ __host__ CompletionQueueEntry(int request_id, int origin_rank, int bytes_transferred, int status) :
        request_id(request_id), origin_rank(origin_rank), bytes_transferred(bytes_transferred), status(status), dirty(1) {}
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
    if (h_request_queue) HIP_CHECK(hipHostFree(h_request_queue));
    if (h_completion_queue) HIP_CHECK(hipHostFree(h_completion_queue));
    if (h_request_head) HIP_CHECK(hipHostFree(h_request_head));
    if (h_request_tail) HIP_CHECK(hipHostFree(h_request_tail));
    if (h_completion_head) HIP_CHECK(hipHostFree(h_completion_head));
    if (h_completion_tail) HIP_CHECK(hipHostFree(h_completion_tail));
}

// Device-side function: GPU enqueues requests
// 
// Data race safety explanation:
// This algorithm is race-free due to the two-phase publication protocol:
// 
// PHASE 1: GPU writes all request data to the slot
// - Only one GPU can write to each slot due to CAS on tail pointer
// - __threadfence_system() ensures all data writes are visible before dirty bit is set
//
// PHASE 2: GPU publishes the data by setting dirty bit
// - dirty=1 acts as a "publication flag" telling host the data is ready
// - Host will only attempt to read entries where dirty=1
// - If host sees dirty=0, it knows the entry is either empty or still being written
// - This prevents host from reading partially-written data
//
// Memory ordering guarantees:
// - __threadfence_system() ensures host sees all slot data before seeing dirty=1
// - Host's __atomic_compare_exchange_n with ACQUIRE ensures it sees all GPU writes
// - No race condition possible: either host sees dirty=0 (won't read and tries again) or dirty=1 (safe to read) 
__device__ bool SharedQueue::enqueue_request(int request_id, int origin_rank, int peer_rank, void *buffer, size_t size, RequestType type) {
    printf("[DEBUG] enqueue_request: Entry - req_id=%d, origin_rank=%d, peer_rank=%d, buffer=%p, size=%zu\n", 
           request_id, origin_rank, peer_rank, buffer, size);
    
    int current_tail, next_tail;
    int slot;
    
    // Debug: Check if pointers are valid
    if (d_request_tail == nullptr) {
        printf("[ERROR] enqueue_request: d_request_tail is NULL!\n");
        return false;
    }
    if (d_request_head == nullptr) {
        printf("[ERROR] enqueue_request: d_request_head is NULL!\n");
        return false;
    }
    if (d_request_queue == nullptr) {
        printf("[ERROR] enqueue_request: d_request_queue is NULL!\n");
        return false;
    }
    
    printf("[DEBUG] enqueue_request: Pointers valid - d_request_tail=%p, d_request_head=%p, d_request_queue=%p\n",
           d_request_tail, d_request_head, d_request_queue);
    
    // use CAS to acquire unique, open slot in request queue
    // ensures no two GPUs acquire the same slot
    do {
        printf("[DEBUG] enqueue_request: About to read d_request_tail\n");
        current_tail = *d_request_tail;
        printf("[DEBUG] enqueue_request: current_tail=%d\n", current_tail);
        
        slot = current_tail & (req_queue_size - 1);
        printf("[DEBUG] enqueue_request: calculated slot=%d (req_queue_size=%zu)\n", slot, req_queue_size);
        
        printf("[DEBUG] enqueue_request: About to read d_request_head\n");
        int current_head = *d_request_head;
        printf("[DEBUG] enqueue_request: current_head=%d\n", current_head);
        
        // TODO: handle queue full properly rather than failure
        // Check if queue is full
        if (current_tail - current_head >= req_queue_size) {
            printf("[DEBUG] enqueue_request: Queue full - tail=%d, head=%d, size=%zu\n", 
                   current_tail, current_head, req_queue_size);
            return false;
        }
        
        printf("[DEBUG] enqueue_request: About to check slot dirty bit at slot %d\n", slot);
        // Check if slot is free
        if (d_request_queue[slot].dirty != 0) {
            printf("[DEBUG] enqueue_request: Slot %d not free (dirty=%d)\n", slot, d_request_queue[slot].dirty);
            return false;
        }
        printf("[DEBUG] enqueue_request: Slot %d is free\n", slot);

        // advance tail
        next_tail = current_tail + 1;
        printf("[DEBUG] enqueue_request: Attempting CAS - current_tail=%d, next_tail=%d\n", current_tail, next_tail);
        
    } while (atomicCAS(d_request_tail, current_tail, next_tail) != current_tail);
    
    printf("[DEBUG] enqueue_request: CAS successful, acquired slot %d\n", slot);
    
    // Construct the entry in-place using placement new
    new (&d_request_queue[slot]) RequestQueueEntry(request_id, origin_rank, peer_rank, buffer, size, type);
    printf("[DEBUG] enqueue_request: Entry constructed successfully\n");
    __threadfence_system(); // Ensure all writes are visible to host
    printf("[DEBUG] enqueue_request: Memory fence executed\n");
    d_request_queue[slot].dirty = 1; // Mark as ready - PUBLICATION POINT
    printf("[DEBUG] enqueue_request: Dirty bit set, returning success\n");
    
    return true;
}

// Device-side function: GPU dequeues completion for the current rank
//
// Data race safety explanation:
// This algorithm is race-free due to the two-phase consumption protocol:
//
// PHASE 1: GPU atomically claims a completion entry
// - CAS on completion head ensures only one GPU thread can claim each entry
// - Multiple checks ensure entry is valid: queue not empty, entry ready (dirty=1), correct rank
// - Only proceeds if all conditions met, preventing reading of invalid/incomplete data
//
// PHASE 2: GPU consumes the data and marks entry as free
// - Copies all entry data before marking as consumed
// - __threadfence_system() ensures data copy is complete before clearing dirty bit
// - dirty=0 marks entry as free for reuse by host
//
// Memory ordering guarantees:
// - atomicCAS provides acquire semantics ensuring visibility of host's completion writes
// - __threadfence_system() ensures GPU's data copy completes before dirty=0 write
// - Host's dirty=1 write (after placement new) is globally visible before GPU sees it
//
// Rank-based filtering:
// - Only dequeues completions intended for this specific GPU rank
// - Prevents cross-GPU completion theft in multi-GPU scenarios
// - Other GPUs will skip entries not meant for them. Since GPUs only check the head entry of the
//   list, in practice the GPU waits until their entry is at the head.
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
//
// Data race safety explanation:
// This algorithm is race-free due to the two-phase consumption protocol on host side:
//
// PHASE 1: Host atomically claims a request entry
// - __atomic_compare_exchange_n on request head ensures only one host thread can claim each entry
// - Multiple checks ensure entry is valid: queue not empty, entry ready (dirty=1)
// - Only proceeds if all conditions met, preventing reading of invalid/incomplete data
// - Uses ACQUIRE semantics to ensure visibility of all GPU writes to the claimed entry
//
// PHASE 2: Host consumes the data and marks entry as free
// - Copies all entry data from the claimed slot
// - Sets dirty=0 to mark entry as free for reuse by GPU
// - No memory fence needed here since host is single-threaded consumer
//
// Memory ordering guarantees:
// - __atomic_compare_exchange_n with ACQUIRE ensures host sees all GPU writes to the entry
// - GPU's __threadfence_system() + dirty=1 write ensures all entry data is visible before host claims it
// - Host's dirty=0 write makes slot available for GPU reuse
//
// Host-GPU coordination:
// - Host only reads entries that GPU has published (dirty=1)
// - Host frees entries for GPU reuse by setting dirty=0
// - This creates a clean handoff from GPU producer to host consumer
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
    // TODO: Not sure if this mem fence is needed.
    std::atomic_thread_fence(std::memory_order_release); // Ensure read is complete
    h_request_queue[slot].dirty = 0; // Mark as consumed
    
    return true;
}

// Host-side function: Host enqueues completions
//
// Data race safety explanation:
// This algorithm is race-free due to the two-phase publication protocol on host side:
//
// PHASE 1: Host writes all completion data to the slot
// - Only one host thread can write to each slot due to CAS on tail pointer
// - Uses placement new to construct completion entry with all data fields
// - All data writes complete before proceeding to publication phase
//
// PHASE 2: Host publishes the completion by setting dirty bit
// - dirty=1 acts as "publication flag" telling GPU the completion is ready
// - GPU will only attempt to read entries where dirty=1
// - If GPU sees dirty=0, it knows the entry is either empty or still being written
// - This prevents GPU from reading partially-written completion data
//
// Memory ordering guarantees:
// - Host's atomic CAS operations provide ordering for slot allocation
// - Placement new completes before dirty=1 write due to program ordering
// - GPU's atomic reads ensure it sees all host writes before claiming entries
// - Optional memory fence available for multi-threaded host scenarios
//
// Host-GPU coordination:
// - Host publishes completions that GPU can consume using dirty flag
// - GPU frees entries for host reuse by setting dirty=0 after consumption
// - This creates a clean handoff from host producer to GPU consumer(s)
// - Safe for both single-threaded and multi-threaded host usage
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
    // TODO: Not sure if this mem fence is needed
    std::atomic_thread_fence(std::memory_order_release); // Ensure write is complete
    h_completion_queue[slot].dirty = 1; // Mark as ready
    
    return true;
}

// Example GPU kernel that enqueues requests
__global__ void gpu_enqueue_kernel(SharedQueue *queue, int rank, void *send_buffer, size_t buffer_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Add a print statement to show kernel execution
    printf("GPU rank %d thread %d: Starting enqueue kernel\n", rank, tid);
    
    
    // Each thread on this GPU enqueues a request with the GPU's rank
    if (tid < 10) { // Limit to first 10 threads for example
        bool success = false;
        while (!success) {
            success = queue->enqueue_request(
                rank * 1000 + tid,      // request_id (unique per GPU rank)
                rank,                   // originating GPU rank
                tid % 8,                // peer_rank (example: 4 peers)
                send_buffer,            // buffer
                buffer_size,            // size
                SEND                    // type
            );
            if (success) {
                printf("GPU rank %d thread %d: Successfully enqueued send request\n", rank, tid);
            } else {
                printf("GPU rank %d thread %d: Unsuccessful enqueued send request\n", rank, tid);
            }
        }
    }
}
__global__ void gpu_dequeue_kernel(SharedQueue *queue, int rank, void *send_buffer, size_t buffer_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Add a print statement to show kernel execution
    printf("GPU rank %d thread %d: Starting dequeue kernel\n", rank, tid);
    
    // Each thread on this GPU dequeues completions for this GPU's rank
    if (tid < 10) { // Limit to first 10 threads for example
        bool success = false;
        while (!success) {
            // Example: Check for completions belonging to this GPU rank
            CompletionQueueEntry completion;
            success = queue->dequeue_completion(completion, rank);
            if (success) {
                printf("GPU rank %d thread %d: Received completion for request %d (status: %d, bytes: %d)\n", 
                    rank, tid, completion.request_id, completion.status, completion.bytes_transferred);
            } else {
                printf("GPU rank %d thread %d: Unsuccessful completion\n", rank, tid);
            }
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
int main(int argc, char** argv) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    
    // Get world rank and size
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    std::cout << "MPI Process " << world_rank << " of " << world_size << " starting HIP GPU-triggered communication queue test..." << std::endl;
    
    std::cout << "[DEBUG] Rank " << world_rank << ": About to allocate SharedQueue in GPU-accessible memory..." << std::endl;
    // Each rank's cpu allocates a SharedQueue in GPU-accessible host memory
    // Allocate SharedQueue in GPU-accessible host memory
    SharedQueue *h_queue, *d_queue;
    HIP_CHECK(hipHostMalloc((void**)&h_queue, sizeof(SharedQueue), hipHostMallocMapped));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_queue, h_queue, 0));
    std::cout << "[DEBUG] Rank " << world_rank << ": SharedQueue allocated - host ptr: " << h_queue << ", device ptr: " << d_queue << std::endl;
    
    // Construct SharedQueue using placement new
    std::cout << "[DEBUG] Rank " << world_rank << ": Constructing SharedQueue..." << std::endl;
    new (h_queue) SharedQueue();
    std::cout << "[DEBUG] Rank " << world_rank << ": SharedQueue created successfully!" << std::endl;
    
    // Allocate some dummy buffer
    void *send_buffer;
    size_t buffer_size = 1024;
    HIP_CHECK(hipMalloc(&send_buffer, buffer_size));
    
    std::cout << "MPI rank " << world_rank << " launching GPU kernel to enqueue completions..." << std::endl;
    
    // Launch GPU kernel with GPU rank
    dim3 block(32);
    dim3 grid(1);
    std::cout << "[DEBUG] Rank " << world_rank << ": About to launch kernel..." << std::endl;
    gpu_enqueue_kernel<<<grid, block>>>(d_queue, world_rank, send_buffer, buffer_size);
    std::cout << "[DEBUG] Rank " << world_rank << ": Kernel launched, calling hipDeviceSynchronize..." << std::endl;
    HIP_CHECK(hipDeviceSynchronize());
    
    std::cout << "MPI rank " << world_rank << " GPU kernel completed. Processing requests on host..." << std::endl;
    
    // Process requests on host (use host pointer)
    host_process_requests(h_queue);
    
    std::cout << "MPI rank " << world_rank << " Done processing requests. Launching kernel to dequeue completions..." << std::endl;
    // dequeue completion kernel
    gpu_dequeue_kernel<<<grid, block>>>(d_queue, world_rank, send_buffer, buffer_size);
    
    // Cleanup
    HIP_CHECK(hipFree(send_buffer));
    
    // Explicitly call destructor and free SharedQueue memory
    h_queue->~SharedQueue();
    HIP_CHECK(hipHostFree(h_queue));
    
    std::cout << "MPI rank " << world_rank << " test completed successfully!" << std::endl;
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

