#ifndef AXON_H
#define AXON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AxonHandle AxonHandle;

/*
 * Open a .axon file. Returns NULL on failure; call axon_last_error for details.
 * The returned handle owns the mmap and must be closed with axon_close.
 */
AxonHandle* axon_open(const char* path);

/*
 * Close a handle returned by axon_open. Any tensor data pointers borrowed from
 * this handle become invalid immediately after this call.
 */
void axon_close(AxonHandle* handle);

uint64_t axon_tensor_count(const AxonHandle* handle);
uint64_t axon_payload_size(const AxonHandle* handle);

/*
 * Copy the model name into buf. Returns bytes written, excluding the trailing
 * NUL. If buf is too small, the string is truncated and NUL-terminated.
 */
uint64_t axon_model_name(const AxonHandle* handle, char* buf, uint64_t buf_size);

/*
 * Copy metadata for the tensor at index. Output pointers may be NULL. If
 * shape_out is non-NULL, the caller must provide enough space for the tensor
 * rank returned via rank_out.
 */
int axon_tensor_info(
    const AxonHandle* handle, uint64_t index,
    char* name_buf, uint64_t name_buf_size,
    uint32_t* dtype_out, uint32_t* rank_out,
    uint64_t* shape_out, uint64_t* data_offset_out,
    uint64_t* data_size_out);

/*
 * Borrow raw tensor data from the mmap. The returned pointer is read-only and is
 * valid only until axon_close is called for the same handle.
 */
const void* axon_tensor_data(const AxonHandle* handle, uint64_t index, uint64_t* data_size);

uint64_t axon_verify_checksums(const AxonHandle* handle, uint64_t* failed_indices, uint64_t* failed_count);
uint64_t axon_version(char* buf, uint64_t buf_size);

/*
 * Copy the last FFI error for this thread. Returns bytes written, excluding the
 * trailing NUL. A return value of 0 means no error is recorded or buf is not
 * writable.
 */
uint64_t axon_last_error(char* buf, uint64_t buf_size);

#define AXON_DTYPE_F32      0
#define AXON_DTYPE_F16      1
#define AXON_DTYPE_BF16     2
#define AXON_DTYPE_I32      3
#define AXON_DTYPE_I64      4
#define AXON_DTYPE_U8       5
#define AXON_DTYPE_Q4       6
#define AXON_DTYPE_Q8       7
#define AXON_DTYPE_F8E4M3   8
#define AXON_DTYPE_F8E5M2   9
#define AXON_DTYPE_I8       10
#define AXON_DTYPE_I16      11

#ifdef __cplusplus
}
#endif

#endif
