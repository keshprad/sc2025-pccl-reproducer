/*
 * Implementation of OFI distributed utility functions
 * Contains definitions for functions extracted from test-common.h (https://github.com/ROCm/aws-ofi-rccl/blob/4a175a74bc69489fc5f001f1abf17386d7bcdf4c/tests/test-common.h)
 * to avoid duplicate symbol issues
 */

#ifndef OFI_DISTRIBUTED_UTILS_H_
#define OFI_DISTRIBUTED_UTILS_H_

#include <nccl_net.h>
#include <rccl/rccl.h>
#include <cstdarg>
#include <cstdint>

// Essential macros from test-common.h
#define STR2(v)		#v
#define STR(v)		STR2(v)

#define OFINCCLCHECK(call) do {							\
	ncclResult_t res = call;						\
	if (res != ncclSuccess) {						\
		NCCL_OFI_WARN("OFI NCCL failure '%s'", ncclGetErrorString(res));	\
		return res;							\
	}									\
} while(false);

#define CUDACHECK(call) do {							\
        hipError_t e = call;							\
        if (e != hipSuccess) {							\
                NCCL_OFI_WARN("Cuda failure '%s'", hipGetErrorString(e));	\
                return ncclUnhandledCudaError;					\
        }									\
} while(false);

// Forward declarations for types from nccl_ofi.h
typedef struct sendComm sendComm_t;
typedef struct recvComm recvComm_t; 
typedef struct listenComm listenComm_t;
typedef struct nccl_ofi_req nccl_ofi_req_t;

// External variables that need to be defined once
extern void (*ofi_log_function)(ncclDebugLogLevel level, unsigned long flags, 
                                const char *file, int line, const char *fmt, ...);
extern pthread_mutex_t nccl_ofi_lock;

// Logging macros
#define NCCL_OFI_WARN(fmt, ...)							\
	(*ofi_log_function)(NCCL_LOG_WARN, NCCL_ALL, __PRETTY_FUNCTION__,	\
	__LINE__, "NET/OFI " fmt, ##__VA_ARGS__)

#define NCCL_OFI_INFO(flags, fmt, ...)				\
	(*ofi_log_function)(NCCL_LOG_INFO, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)

#define NCCL_OFI_TRACE(flags, fmt, ...)				\
	(*ofi_log_function)(NCCL_LOG_TRACE, flags,		\
	__PRETTY_FUNCTION__, __LINE__, "NET/OFI " fmt,		\
	##__VA_ARGS__)

// Function declarations
void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
            int line, const char *fmt, ...);

ncclNet_t* get_extNet(void);

#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
void print_dev_props(int dev, ncclNetProperties_t *props);
#endif

int is_gdr_supported_nic(uint64_t ptr_support);

ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type);
ncclResult_t deallocate_buffer(void *buf, int buffer_type);
ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type);
ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type);

#endif // OFI_DISTRIBUTED_UTILS_H_
