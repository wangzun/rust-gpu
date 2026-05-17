//! Shader-side descriptor-indexing storage-buffer accessors.
//!
//! The functions in this module are only executable for `target_arch = "spirv"`.
//! On CPU targets they panic so normal editor tooling and `cargo check` can still
//! type-check shader crates.

#[cfg(target_arch = "spirv")]
use core::arch::asm;
#[cfg(target_arch = "spirv")]
use core::mem::MaybeUninit;

/// Loads one element from a storage buffer descriptor in set `0`, binding `0`.
///
/// # Safety
///
/// `id` must identify a live storage buffer descriptor containing `index`, and
/// the element at `index` must have the layout of `T`.
#[inline]
pub unsafe fn storage_buffer_load_from_u32<T: Copy>(id: u32, index: u32) -> T {
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = id;
        let _ = index;
        unimplemented!("`storage_buffer_load_from_u32` is only available on SPIR-V targets")
    }

    #[cfg(target_arch = "spirv")]
    unsafe {
        let mut result = MaybeUninit::<T>::uninit();
        storage_buffer_load_from_u32_inner(id, index, result.as_mut_ptr());
        result.assume_init()
    }
}

/// Stores one element into a storage buffer descriptor in set `0`, binding `0`.
///
/// # Safety
///
/// `id` must identify a live writable storage buffer descriptor containing
/// `index`, and the element at `index` must have the layout of `T`.
#[inline]
pub unsafe fn storage_buffer_store_from_u32<T: Copy>(id: u32, index: u32, value: T) {
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = id;
        let _ = index;
        let _ = value;
        unimplemented!("`storage_buffer_store_from_u32` is only available on SPIR-V targets")
    }

    #[cfg(target_arch = "spirv")]
    unsafe {
        storage_buffer_store_from_u32_inner(id, index, &value);
    }
}

#[cfg(target_arch = "spirv")]
#[inline]
unsafe fn storage_buffer_load_from_u32_inner<T>(id: u32, index: u32, result: *mut T) {
    unsafe {
        asm!(
            "OpCapability RuntimeDescriptorArray",
            "OpExtension \"SPV_EXT_descriptor_indexing\"",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpDecorate %dst_heap_resource_handles Binding 0",
            "OpDecorate %dst_heap_resource_handles DescriptorSet 0",
            "OpDecorate %dst_heap_value_runtime_array ArrayStride {stride}",
            "%dst_heap_u32 = OpTypeInt 32 0",
            "%dst_heap_u32_0 = OpConstant %dst_heap_u32 0",
            "%dst_heap_index = OpLoad %dst_heap_u32 {id}",
            "%dst_heap_element_index = OpLoad %dst_heap_u32 {index}",
            "%dst_heap_value_runtime_array = OpTypeRuntimeArray typeof*{result}",
            "%dst_heap_storage_buffer = OpTypeStruct %dst_heap_value_runtime_array",
            "%dst_heap_storage_buffer_array = OpTypeRuntimeArray %dst_heap_storage_buffer",
            "%dst_heap_resource_handles_ptr_type = OpTypePointer Generic %dst_heap_storage_buffer_array",
            "%dst_heap_storage_buffer_ptr_type = OpTypePointer Generic %dst_heap_storage_buffer",
            "%dst_heap_value_ptr_type = OpTypePointer Generic typeof*{result}",
            "%dst_heap_resource_handles = OpVariable %dst_heap_resource_handles_ptr_type StorageBuffer",
            "%dst_heap_buffer_ptr = OpAccessChain %dst_heap_storage_buffer_ptr_type %dst_heap_resource_handles %dst_heap_index",
            "%dst_heap_element_ptr = OpAccessChain %dst_heap_value_ptr_type %dst_heap_buffer_ptr %dst_heap_u32_0 %dst_heap_element_index",
            "%dst_heap_value = OpLoad typeof*{result} %dst_heap_element_ptr",
            "OpStore {result} %dst_heap_value",
            id = in(reg) &id,
            index = in(reg) &index,
            result = in(reg) result,
            stride = const core::mem::size_of::<T>(),
        );
    }
}

#[cfg(target_arch = "spirv")]
#[inline]
unsafe fn storage_buffer_store_from_u32_inner<T>(id: u32, index: u32, value: *const T) {
    unsafe {
        asm!(
            "OpCapability RuntimeDescriptorArray",
            "OpExtension \"SPV_EXT_descriptor_indexing\"",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpDecorate %dst_heap_resource_handles Binding 0",
            "OpDecorate %dst_heap_resource_handles DescriptorSet 0",
            "OpDecorate %dst_heap_value_runtime_array ArrayStride {stride}",
            "%dst_heap_u32 = OpTypeInt 32 0",
            "%dst_heap_u32_0 = OpConstant %dst_heap_u32 0",
            "%dst_heap_index = OpLoad %dst_heap_u32 {id}",
            "%dst_heap_element_index = OpLoad %dst_heap_u32 {index}",
            "%dst_heap_value_runtime_array = OpTypeRuntimeArray typeof*{value}",
            "%dst_heap_storage_buffer = OpTypeStruct %dst_heap_value_runtime_array",
            "%dst_heap_storage_buffer_array = OpTypeRuntimeArray %dst_heap_storage_buffer",
            "%dst_heap_resource_handles_ptr_type = OpTypePointer Generic %dst_heap_storage_buffer_array",
            "%dst_heap_storage_buffer_ptr_type = OpTypePointer Generic %dst_heap_storage_buffer",
            "%dst_heap_value_ptr_type = OpTypePointer Generic typeof*{value}",
            "%dst_heap_resource_handles = OpVariable %dst_heap_resource_handles_ptr_type StorageBuffer",
            "%dst_heap_buffer_ptr = OpAccessChain %dst_heap_storage_buffer_ptr_type %dst_heap_resource_handles %dst_heap_index",
            "%dst_heap_element_ptr = OpAccessChain %dst_heap_value_ptr_type %dst_heap_buffer_ptr %dst_heap_u32_0 %dst_heap_element_index",
            "%dst_heap_value = OpLoad typeof*{value} {value}",
            "OpStore %dst_heap_element_ptr %dst_heap_value",
            id = in(reg) &id,
            index = in(reg) &index,
            value = in(reg) value,
            stride = const core::mem::size_of::<T>(),
        );
    }
}
