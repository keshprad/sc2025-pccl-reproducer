/*
 * Implementation of OFI distributed utility functions
 * Contains definitions for functions extracted from test-common.h (https://github.com/ROCm/aws-ofi-rccl/blob/4a175a74bc69489fc5f001f1abf17386d7bcdf4c/tests/test-common.h)
 * to avoid duplicate symbol issues
 */

#include "ofi_distributed_utils.h"
#include <nccl_ofi_param.h>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <pthread.h>

// Global variable definitions (defined once here to avoid duplicates)
void (*ofi_log_function)(ncclDebugLogLevel level, unsigned long flags, 
                        const char *file, int line, const char *fmt, ...) = NULL;
pthread_mutex_t nccl_ofi_lock = PTHREAD_MUTEX_INITIALIZER;

// Logger function implementation
void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
	    int line, const char *fmt, ...)
{
	va_list vargs;

	switch (level) {
		case NCCL_LOG_WARN:
#if OFI_NCCL_WARN
            printf("WARN: Function: %s Line: %d: ", filefunc, line);
#endif
			break;
		case NCCL_LOG_INFO:
#if OFI_NCCL_INFO
            printf("INFO: Function: %s Line: %d: ", filefunc, line);
#endif
			break;
		case NCCL_LOG_TRACE:
#if OFI_NCCL_TRACE
			printf("TRACE: Function: %s Line: %d: ", filefunc, line);
#endif
			break;
		default:
			break;
	};

	va_start(vargs, fmt);
	vprintf(fmt, vargs);
	printf("\n");
	va_end(vargs);
}

// Get external network plugin
ncclNet_t *get_extNet(void)
{
	void *netPluginLib = NULL;
	ncclNet_t *extNet = NULL;

	netPluginLib = dlopen("librccl-net.so", RTLD_NOW | RTLD_LOCAL);
	if (netPluginLib == NULL) {
		NCCL_OFI_WARN("Unable to load librccl-net.so: %s", dlerror());
		return NULL;
	}

	extNet = (ncclNet_t *)dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
	if (extNet == NULL) {
		NCCL_OFI_WARN("NetPlugin, could not find %s symbol",
			      STR(NCCL_PLUGIN_SYMBOL));
	}

	return extNet;
}

// Print device properties
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 6, 4))
void print_dev_props(int dev, ncclNetProperties_t *props)
{
        NCCL_OFI_TRACE(NCCL_NET, "****************** Device %d Properties ******************", dev);
        NCCL_OFI_TRACE(NCCL_NET, "%s: PCIe Path: %s", props->name, props->pciPath);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Plugin Support: %d", props->name, props->ptrSupport);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device GUID: %d", props->name, props->guid);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Speed: %d", props->name, props->speed);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Port: %d", props->name, props->port);
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Communicators: %d", props->name, props->maxComms);
#if (NCCL_VERSION_CODE >= NCCL_VERSION(2, 12, 0))
        NCCL_OFI_TRACE(NCCL_NET, "%s: Device Maximum Grouped Receives: %d", props->name, props->maxRecvs);
#endif
}
#endif

// Check if NIC supports GDR
int is_gdr_supported_nic(uint64_t ptr_support)
{
	if (ptr_support & NCCL_PTR_CUDA)
		return 1;

	return 0;
}

// Buffer allocation function
ncclResult_t allocate_buff(void **buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating CUDA buffer");
		CUDACHECK(hipExtMallocWithFlags(buf, size, hipDeviceMallocFinegrained));
		break;
	case NCCL_PTR_HOST:
		NCCL_OFI_TRACE(NCCL_NET, "Allocating host buffer");
		CUDACHECK(hipHostMalloc((void **)buf, size, hipHostMallocMapped));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

// Buffer deallocation function
ncclResult_t deallocate_buffer(void *buf, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(hipFree((void *)buf));
		break;
	case NCCL_PTR_HOST:
		CUDACHECK(hipHostFree((void *)buf));
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

// Buffer initialization function
ncclResult_t initialize_buff(void *buf, size_t size, int buffer_type)
{
	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		CUDACHECK(hipMemset(buf, '1', size));
		break;
	case NCCL_PTR_HOST:
		memset(buf, '1', size);
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return ncclSuccess;
}

// Data validation function
ncclResult_t validate_data(char *recv_buf, char *expected_buf, size_t size, int buffer_type)
{
	int ret = 0;
	char *host_recv_buf = NULL;

	switch (buffer_type) {
	case NCCL_PTR_CUDA:
		NCCL_OFI_TRACE(NCCL_NET, "Validating CUDA buffer");
		OFINCCLCHECK(allocate_buff((void **)&host_recv_buf, size, NCCL_PTR_HOST));
		CUDACHECK(hipMemcpy(host_recv_buf, recv_buf, size, hipMemcpyDeviceToHost));
		ret = memcmp(host_recv_buf, expected_buf, size);
		OFINCCLCHECK(deallocate_buffer(host_recv_buf, NCCL_PTR_HOST));
		break;
	case NCCL_PTR_HOST:
		NCCL_OFI_TRACE(NCCL_NET, "Validating host buffer");
		ret = memcmp(recv_buf, expected_buf, size);
		break;
	default:
		NCCL_OFI_WARN("Unidentified buffer type: %d", buffer_type);
		return ncclInvalidArgument;
	}

	return (ret == 0) ? ncclSuccess : ncclInvalidArgument;
}
