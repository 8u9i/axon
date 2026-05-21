use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use axon_core::*;

pub struct AxonHandle {
    file: MappedAxonFile<'static>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl Into<String>) {
    let message = message.into().replace('\0', "\\0");
    let c_message = CString::new(message).unwrap_or_default();
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(c_message);
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn copy_c_string(value: &CString, buf: *mut c_char, buf_size: u64) -> u64 {
    if buf.is_null() || buf_size == 0 {
        return 0;
    }

    let bytes = value.as_bytes_with_nul();
    let to_copy = bytes.len().min(buf_size as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, to_copy);
        if to_copy == buf_size as usize {
            *buf.add(buf_size as usize - 1) = 0;
        }
    }
    to_copy.saturating_sub(1) as u64
}

/// Open an `.axon` file and return an opaque handle.
///
/// # Safety
///
/// `path` must be a valid, null-terminated UTF-8 C string. The returned handle
/// must be released exactly once with `axon_close`. On failure this returns
/// null and records a message retrievable via `axon_last_error`.
#[no_mangle]
pub unsafe extern "C" fn axon_open(path: *const c_char) -> *mut AxonHandle {
    if path.is_null() {
        set_last_error("axon_open received a null path");
        return ptr::null_mut();
    }

    let cstr = unsafe { CStr::from_ptr(path) };
    let path_str = match cstr.to_str() {
        Ok(s) => s,
        Err(err) => {
            set_last_error(format!("path is not valid UTF-8: {err}"));
            return ptr::null_mut();
        }
    };

    match MappedAxonFile::open(Path::new(path_str)) {
        Ok(file) => {
            clear_last_error();
            Box::into_raw(Box::new(AxonHandle { file }))
        }
        Err(err) => {
            set_last_error(format!("failed to open {path_str}: {err}"));
            ptr::null_mut()
        }
    }
}

/// Close a handle returned by `axon_open`.
///
/// # Safety
///
/// `handle` must either be null or a pointer returned by `axon_open` that has
/// not already been closed. After this call, all tensor data pointers borrowed
/// from the handle are invalid.
#[no_mangle]
pub unsafe extern "C" fn axon_close(handle: *mut AxonHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Return the number of tensors in an open file, or 0 for a null handle.
///
/// # Safety
///
/// `handle` must be null or a live handle returned by `axon_open`.
#[no_mangle]
pub unsafe extern "C" fn axon_tensor_count(handle: *const AxonHandle) -> u64 {
    if handle.is_null() {
        set_last_error("axon_tensor_count received a null handle");
        return 0;
    }
    clear_last_error();
    unsafe { &*handle }.file.file.header.tensor_count
}

/// Return the total payload size in bytes, or 0 for a null handle.
///
/// # Safety
///
/// `handle` must be null or a live handle returned by `axon_open`.
#[no_mangle]
pub unsafe extern "C" fn axon_payload_size(handle: *const AxonHandle) -> u64 {
    if handle.is_null() {
        set_last_error("axon_payload_size received a null handle");
        return 0;
    }
    clear_last_error();
    unsafe { &*handle }.file.file.header.payload_size
}

/// Copy the model name into `buf` and return the number of bytes written,
/// excluding the terminating null byte.
///
/// # Safety
///
/// `handle` must be a live handle. `buf` must point to at least `buf_size`
/// writable bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn axon_model_name(
    handle: *const AxonHandle,
    buf: *mut c_char,
    buf_size: u64,
) -> u64 {
    if handle.is_null() {
        set_last_error("axon_model_name received a null handle");
        return 0;
    }

    let model = unsafe { &*handle }
        .file
        .file
        .manifest
        .model
        .as_deref()
        .unwrap_or("");
    let cstr = CString::new(model).unwrap_or_default();
    clear_last_error();
    copy_c_string(&cstr, buf, buf_size)
}

/// Copy tensor metadata for `index` into the provided output pointers.
///
/// Returns 1 on success and 0 on failure.
///
/// # Safety
///
/// `handle` must be a live handle. All output pointers may be null. Non-null
/// buffers must point to writable memory large enough for the requested data.
/// `shape_out`, when non-null, must have room for `rank_out` dimensions.
#[no_mangle]
pub unsafe extern "C" fn axon_tensor_info(
    handle: *const AxonHandle,
    index: u64,
    name_buf: *mut c_char,
    name_buf_size: u64,
    dtype_out: *mut u32,
    rank_out: *mut u32,
    shape_out: *mut u64,
    data_offset_out: *mut u64,
    data_size_out: *mut u64,
) -> i32 {
    if handle.is_null() {
        set_last_error("axon_tensor_info received a null handle");
        return 0;
    }

    let handle = unsafe { &*handle };
    let order = &handle.file.file.manifest.tensor_order;
    if index as usize >= order.len() {
        set_last_error(format!("tensor index {index} is out of range"));
        return 0;
    }

    let name = &order[index as usize];
    let desc = match handle.file.file.manifest.get_tensor(name) {
        Some(d) => d,
        None => {
            set_last_error(format!("tensor descriptor missing for {name}"));
            return 0;
        }
    };

    let cstr = CString::new(desc.name_str()).unwrap_or_default();
    copy_c_string(&cstr, name_buf, name_buf_size);

    if !dtype_out.is_null() {
        unsafe {
            *dtype_out = desc.dtype;
        }
    }
    if !rank_out.is_null() {
        unsafe {
            *rank_out = desc.rank;
        }
    }
    if !shape_out.is_null() {
        unsafe {
            for i in 0..desc.rank as usize {
                *shape_out.add(i) = desc.shape[i];
            }
        }
    }
    if !data_offset_out.is_null() {
        unsafe {
            *data_offset_out = desc.data_offset;
        }
    }
    if !data_size_out.is_null() {
        unsafe {
            *data_size_out = desc.data_size;
        }
    }

    clear_last_error();
    1
}

/// Borrow raw tensor bytes for `index`.
///
/// Returns a pointer into the memory-mapped file, or null on failure.
///
/// # Safety
///
/// `handle` must be a live handle. The returned pointer remains valid only
/// while the handle is open and must not be written to or freed by the caller.
/// `data_size` may be null; when non-null it must be writable.
#[no_mangle]
pub unsafe extern "C" fn axon_tensor_data(
    handle: *const AxonHandle,
    index: u64,
    data_size: *mut u64,
) -> *const u8 {
    if handle.is_null() {
        set_last_error("axon_tensor_data received a null handle");
        return ptr::null();
    }

    let handle = unsafe { &*handle };
    let order = &handle.file.file.manifest.tensor_order;
    if index as usize >= order.len() {
        set_last_error(format!("tensor index {index} is out of range"));
        return ptr::null();
    }

    let name = &order[index as usize];
    let desc = match handle.file.file.manifest.get_tensor(name) {
        Some(d) => d,
        None => {
            set_last_error(format!("tensor descriptor missing for {name}"));
            return ptr::null();
        }
    };

    if !data_size.is_null() {
        unsafe {
            *data_size = desc.data_size;
        }
    }

    let offset = desc.data_offset as usize;
    let Some(end) = offset.checked_add(desc.data_size as usize) else {
        set_last_error(format!("tensor {name} byte range overflows usize"));
        return ptr::null();
    };

    if end <= handle.file.file.data.len() {
        clear_last_error();
        handle.file.file.data[offset..end].as_ptr()
    } else {
        set_last_error(format!("tensor {name} byte range exceeds file size"));
        ptr::null()
    }
}

/// Verify all tensor checksums and optionally copy failed tensor indices.
///
/// Returns the number of valid tensors.
///
/// # Safety
///
/// `handle` must be a live handle. `failed_indices`, when non-null, must have
/// enough capacity for every failed tensor. `failed_count` may be null.
#[no_mangle]
pub unsafe extern "C" fn axon_verify_checksums(
    handle: *const AxonHandle,
    failed_indices: *mut u64,
    failed_count: *mut u64,
) -> u64 {
    if handle.is_null() {
        set_last_error("axon_verify_checksums received a null handle");
        return 0;
    }

    let handle = unsafe { &*handle };
    let results = handle.file.file.verify_all_checksums();
    let mut valid = 0u64;
    let mut failed_list = Vec::new();

    for (name, ok) in &results {
        if *ok {
            valid += 1;
        } else if let Some(pos) = handle
            .file
            .file
            .manifest
            .tensor_order
            .iter()
            .position(|n| n == name)
        {
            failed_list.push(pos as u64);
        }
    }

    if !failed_indices.is_null() && !failed_list.is_empty() {
        unsafe {
            for (i, &idx) in failed_list.iter().enumerate() {
                *failed_indices.add(i) = idx;
            }
        }
    }
    if !failed_count.is_null() {
        unsafe {
            *failed_count = failed_list.len() as u64;
        }
    }

    clear_last_error();
    valid
}

/// Copy the Axon format version string into `buf`.
///
/// # Safety
///
/// `buf` must point to at least `buf_size` writable bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn axon_version(buf: *mut c_char, buf_size: u64) -> u64 {
    let version = CString::new(format!("Axon v{}", axon_core::AXON_VERSION)).unwrap_or_default();
    copy_c_string(&version, buf, buf_size)
}

/// Copy the last FFI error for this thread into `buf`.
///
/// Returns the number of bytes copied, excluding the terminating null byte. A
/// return value of 0 means no error is currently recorded or the buffer was not
/// writable.
///
/// # Safety
///
/// `buf` must point to at least `buf_size` writable bytes when non-null.
#[no_mangle]
pub unsafe extern "C" fn axon_last_error(buf: *mut c_char, buf_size: u64) -> u64 {
    LAST_ERROR.with(|slot| {
        let borrowed = slot.borrow();
        match borrowed.as_ref() {
            Some(message) => copy_c_string(message, buf, buf_size),
            None => 0,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_open_sets_last_error() {
        let handle = unsafe { axon_open(std::ptr::null()) };
        assert!(handle.is_null());

        let mut buf = [0i8; 128];
        let len = unsafe { axon_last_error(buf.as_mut_ptr(), buf.len() as u64) };
        assert!(len > 0);
    }

    #[test]
    fn version_copies_into_small_buffer_safely() {
        let mut buf = [0i8; 4];
        let len = unsafe { axon_version(buf.as_mut_ptr(), buf.len() as u64) };
        assert_eq!(len, 3);
        assert_eq!(buf[3], 0);
    }
}
