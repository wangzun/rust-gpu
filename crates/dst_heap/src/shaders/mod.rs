//! Shader-side descriptor-indexing storage-buffer accessors.
//!
//! The functions in this module are only executable for `target_arch = "spirv"`.
//! On CPU targets they panic so normal editor tooling and `cargo check` can still
//! type-check shader crates.

#[cfg(target_arch = "spirv")]
use core::arch::asm;
use core::marker::PhantomData;
#[cfg(target_arch = "spirv")]
use core::mem::MaybeUninit;

#[cfg(target_arch = "spirv")]
use spirv_std::VectorTruncateInto;
use spirv_std::{
    Float, Integer, Sampler, TypedBuffer,
    image::{Image, ImageCoordinate, Multisampled, SampleType, Sampled},
};

/// Descriptor set used by the Pumicite mutable resource heap.
pub const RESOURCE_DESCRIPTOR_SET: u32 = 0;

/// Binding used by the Pumicite mutable resource heap.
pub const RESOURCE_BINDING: u32 = 0;

/// Descriptor set used by the Pumicite sampler heap.
pub const SAMPLER_DESCRIPTOR_SET: u32 = 1;

/// Binding used by the Pumicite sampler heap.
pub const SAMPLER_BINDING: u32 = 0;

/// Bindless image descriptor handle.
///
/// On CPU targets this keeps the requested pointer-shaped API for type-checking.
/// On SPIR-V targets it stores the descriptor index instead, because rust-gpu
/// cannot currently lower Rust locals containing pointers to opaque image types.
#[repr(transparent)]
pub struct BindlessImage<I> {
    #[cfg(not(target_arch = "spirv"))]
    #[allow(dead_code)]
    ptr: *const I,
    #[cfg(target_arch = "spirv")]
    id: u32,
    #[allow(dead_code)]
    _marker: PhantomData<*const I>,
}

impl<I> Copy for BindlessImage<I> {}

impl<I> Clone for BindlessImage<I> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Bindless sampler descriptor handle.
#[repr(transparent)]
pub struct BindlessSampler {
    #[cfg(not(target_arch = "spirv"))]
    #[allow(dead_code)]
    ptr: *const Sampler,
    #[cfg(target_arch = "spirv")]
    id: u32,
}

impl Copy for BindlessSampler {}

impl Clone for BindlessSampler {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

/// Gets a typed storage-buffer descriptor from set `0`, binding `0`.
#[inline(always)]
pub fn storage_buffer_from_u32<T: 'static>(id: u32) -> &'static mut TypedBuffer<[T]> {
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = id;
        unimplemented!("`storage_buffer_from_u32` is only available on SPIR-V targets")
    }

    #[cfg(target_arch = "spirv")]
    unsafe {
        let mut result = MaybeUninit::<&mut TypedBuffer<[T]>>::uninit();
        let mut value_type = MaybeUninit::<T>::uninit();
        storage_buffer_from_u32_inner(id, value_type.as_mut_ptr(), result.as_mut_ptr());
        result.assume_init()
    }
}

/// Gets an image descriptor handle from set `0`, binding `0`.
///
/// `I` should be a `spirv_std::Image!` type, for example
/// `spirv_std::image::Image2d` or `spirv_std::image::StorageImage2d`.
#[inline(always)]
pub fn image_from_u32<I: 'static + Sized + Send + Sync>(id: u32) -> BindlessImage<I> {
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = id;
        unimplemented!("`image_from_u32` is only available on SPIR-V targets")
    }

    #[cfg(target_arch = "spirv")]
    {
        BindlessImage {
            id,
            _marker: PhantomData,
        }
    }
}

/// Gets a sampler descriptor handle from set `1`, binding `0`.
#[inline(always)]
pub fn sampler_from_u32(id: u32) -> BindlessSampler {
    #[cfg(not(target_arch = "spirv"))]
    {
        let _ = id;
        unimplemented!("`sampler_from_u32` is only available on SPIR-V targets")
    }

    #[cfg(target_arch = "spirv")]
    {
        BindlessSampler { id }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    BindlessImage<
        Image<
            SampledType,
            DIM,
            DEPTH,
            ARRAYED,
            { Multisampled::False as u32 },
            { Sampled::Yes as u32 },
            FORMAT,
            COMPONENTS,
        >,
    >
{
    /// Samples a bindless sampled image with a bindless sampler.
    #[inline(always)]
    pub fn sample<F>(
        &self,
        sampler: BindlessSampler,
        coord: impl ImageCoordinate<F, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        F: Float,
    {
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = (sampler, coord);
            unimplemented!("`BindlessImage::sample` is only available on SPIR-V targets")
        }

        #[cfg(target_arch = "spirv")]
        unsafe {
            let mut result = SampledType::Vec4::default();
            let mut image_type = MaybeUninit::<
                Image<
                    SampledType,
                    DIM,
                    DEPTH,
                    ARRAYED,
                    { Multisampled::False as u32 },
                    { Sampled::Yes as u32 },
                    FORMAT,
                    COMPONENTS,
                >,
            >::uninit();
            let mut sampler_type = MaybeUninit::<Sampler>::uninit();
            asm!(
                "OpCapability RuntimeDescriptorArray",
                "OpExtension \"SPV_EXT_descriptor_indexing\"",
                "OpDecorate %dst_heap_image_handles Binding {resource_binding}",
                "OpDecorate %dst_heap_image_handles DescriptorSet {resource_set}",
                "OpDecorate %dst_heap_sampler_handles Binding {sampler_binding}",
                "OpDecorate %dst_heap_sampler_handles DescriptorSet {sampler_set}",
                "OpDecorate %image NonUniform",
                "OpDecorate %sampler NonUniform",
                "OpDecorate %sampledImage NonUniform",
                "OpDecorate %result NonUniform",
                "%dst_heap_u32 = OpTypeInt 32 0",
                "%dst_heap_image_index = OpLoad %dst_heap_u32 {image_id}",
                "%dst_heap_sampler_index = OpLoad %dst_heap_u32 {sampler_id}",
                "OpDecorate %dst_heap_image_index NonUniform",
                "OpDecorate %dst_heap_sampler_index NonUniform",
                "%dst_heap_image_array = OpTypeRuntimeArray typeof*{image_type}",
                "%dst_heap_image_handles_ptr_type = OpTypePointer Generic %dst_heap_image_array",
                "%dst_heap_image_ptr_type = OpTypePointer Generic typeof*{image_type}",
                "%dst_heap_image_handles = OpVariable %dst_heap_image_handles_ptr_type UniformConstant",
                "%dst_heap_image_ptr = OpAccessChain %dst_heap_image_ptr_type %dst_heap_image_handles %dst_heap_image_index",
                "%dst_heap_sampler_array = OpTypeRuntimeArray typeof*{sampler_type}",
                "%dst_heap_sampler_handles_ptr_type = OpTypePointer Generic %dst_heap_sampler_array",
                "%dst_heap_sampler_ptr_type = OpTypePointer Generic typeof*{sampler_type}",
                "%dst_heap_sampler_handles = OpVariable %dst_heap_sampler_handles_ptr_type UniformConstant",
                "%dst_heap_sampler_ptr = OpAccessChain %dst_heap_sampler_ptr_type %dst_heap_sampler_handles %dst_heap_sampler_index",
                "%image = OpLoad typeof*{image_type} %dst_heap_image_ptr",
                "%sampler = OpLoad typeof*{sampler_type} %dst_heap_sampler_ptr",
                "%coord = OpLoad _ {coord}",
                "%sampledImage = OpSampledImage _ %image %sampler",
                "%result = OpImageSampleImplicitLod typeof*{result} %sampledImage %coord",
                "OpStore {result} %result",
                result = in(reg) &mut result,
                image_id = in(reg) &self.id,
                sampler_id = in(reg) &sampler.id,
                image_type = in(reg) image_type.as_mut_ptr(),
                sampler_type = in(reg) sampler_type.as_mut_ptr(),
                coord = in(reg) &coord,
                resource_set = const RESOURCE_DESCRIPTOR_SET,
                resource_binding = const RESOURCE_BINDING,
                sampler_set = const SAMPLER_DESCRIPTOR_SET,
                sampler_binding = const SAMPLER_BINDING,
            );
            result.truncate_into()
        }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    BindlessImage<
        Image<
            SampledType,
            DIM,
            DEPTH,
            ARRAYED,
            MULTISAMPLED,
            { Sampled::No as u32 },
            FORMAT,
            COMPONENTS,
        >,
    >
{
    /// Reads one texel from a bindless storage image.
    #[inline(always)]
    pub fn read<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = coordinate;
            unimplemented!("`BindlessImage::read` is only available on SPIR-V targets")
        }

        #[cfg(target_arch = "spirv")]
        unsafe {
            let mut result = SampledType::Vec4::default();
            let mut image_type = MaybeUninit::<
                Image<
                    SampledType,
                    DIM,
                    DEPTH,
                    ARRAYED,
                    MULTISAMPLED,
                    { Sampled::No as u32 },
                    FORMAT,
                    COMPONENTS,
                >,
            >::uninit();
            asm!(
                "OpCapability RuntimeDescriptorArray",
                "OpCapability StorageImageReadWithoutFormat",
                "OpExtension \"SPV_EXT_descriptor_indexing\"",
                "OpDecorate %dst_heap_image_handles Binding {binding}",
                "OpDecorate %dst_heap_image_handles DescriptorSet {set}",
                "OpDecorate %image NonUniform",
                "OpDecorate %result NonUniform",
                "%dst_heap_u32 = OpTypeInt 32 0",
                "%dst_heap_image_index = OpLoad %dst_heap_u32 {image_id}",
                "OpDecorate %dst_heap_image_index NonUniform",
                "%dst_heap_image_array = OpTypeRuntimeArray typeof*{image_type}",
                "%dst_heap_image_handles_ptr_type = OpTypePointer Generic %dst_heap_image_array",
                "%dst_heap_image_ptr_type = OpTypePointer Generic typeof*{image_type}",
                "%dst_heap_image_handles = OpVariable %dst_heap_image_handles_ptr_type UniformConstant",
                "%dst_heap_image_ptr = OpAccessChain %dst_heap_image_ptr_type %dst_heap_image_handles %dst_heap_image_index",
                "%image = OpLoad typeof*{image_type} %dst_heap_image_ptr",
                "%coordinate = OpLoad _ {coordinate}",
                "%result = OpImageRead typeof*{result} %image %coordinate",
                "OpStore {result} %result",
                image_id = in(reg) &self.id,
                image_type = in(reg) image_type.as_mut_ptr(),
                coordinate = in(reg) &coordinate,
                result = in(reg) &mut result,
                set = const RESOURCE_DESCRIPTOR_SET,
                binding = const RESOURCE_BINDING,
            );
            result.truncate_into()
        }
    }

    /// Writes one texel to a bindless storage image.
    #[inline(always)]
    pub unsafe fn write<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
        texels: SampledType::SampleResult,
    ) where
        I: Integer,
    {
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = (coordinate, texels);
            unimplemented!("`BindlessImage::write` is only available on SPIR-V targets")
        }

        #[cfg(target_arch = "spirv")]
        unsafe {
            let mut image_type = MaybeUninit::<
                Image<
                    SampledType,
                    DIM,
                    DEPTH,
                    ARRAYED,
                    MULTISAMPLED,
                    { Sampled::No as u32 },
                    FORMAT,
                    COMPONENTS,
                >,
            >::uninit();
            asm!(
                "OpCapability RuntimeDescriptorArray",
                "OpCapability StorageImageWriteWithoutFormat",
                "OpExtension \"SPV_EXT_descriptor_indexing\"",
                "OpDecorate %dst_heap_image_handles Binding {binding}",
                "OpDecorate %dst_heap_image_handles DescriptorSet {set}",
                "OpDecorate %image NonUniform",
                "%dst_heap_u32 = OpTypeInt 32 0",
                "%dst_heap_image_index = OpLoad %dst_heap_u32 {image_id}",
                "OpDecorate %dst_heap_image_index NonUniform",
                "%dst_heap_image_array = OpTypeRuntimeArray typeof*{image_type}",
                "%dst_heap_image_handles_ptr_type = OpTypePointer Generic %dst_heap_image_array",
                "%dst_heap_image_ptr_type = OpTypePointer Generic typeof*{image_type}",
                "%dst_heap_image_handles = OpVariable %dst_heap_image_handles_ptr_type UniformConstant",
                "%dst_heap_image_ptr = OpAccessChain %dst_heap_image_ptr_type %dst_heap_image_handles %dst_heap_image_index",
                "%image = OpLoad typeof*{image_type} %dst_heap_image_ptr",
                "%coordinate = OpLoad _ {coordinate}",
                "%texels = OpLoad _ {texels}",
                "OpImageWrite %image %coordinate %texels",
                image_id = in(reg) &self.id,
                image_type = in(reg) image_type.as_mut_ptr(),
                coordinate = in(reg) &coordinate,
                texels = in(reg) &texels,
                set = const RESOURCE_DESCRIPTOR_SET,
                binding = const RESOURCE_BINDING,
            );
        }
    }
}

impl<
    SampledType: SampleType<FORMAT, COMPONENTS>,
    const DIM: u32,
    const DEPTH: u32,
    const ARRAYED: u32,
    const MULTISAMPLED: u32,
    const FORMAT: u32,
    const COMPONENTS: u32,
>
    BindlessImage<
        Image<
            SampledType,
            DIM,
            DEPTH,
            ARRAYED,
            MULTISAMPLED,
            { Sampled::Unknown as u32 },
            FORMAT,
            COMPONENTS,
        >,
    >
{
    /// Reads one texel from a bindless image with unknown sampledness.
    #[inline(always)]
    pub fn read<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> SampledType::SampleResult
    where
        I: Integer,
    {
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = coordinate;
            unimplemented!("`BindlessImage::read` is only available on SPIR-V targets")
        }

        #[cfg(target_arch = "spirv")]
        unsafe {
            let mut result = SampledType::Vec4::default();
            let mut image_type = MaybeUninit::<
                Image<
                    SampledType,
                    DIM,
                    DEPTH,
                    ARRAYED,
                    MULTISAMPLED,
                    { Sampled::Unknown as u32 },
                    FORMAT,
                    COMPONENTS,
                >,
            >::uninit();
            asm!(
                "OpCapability RuntimeDescriptorArray",
                "OpCapability StorageImageReadWithoutFormat",
                "OpExtension \"SPV_EXT_descriptor_indexing\"",
                "OpDecorate %dst_heap_image_handles Binding {binding}",
                "OpDecorate %dst_heap_image_handles DescriptorSet {set}",
                "OpDecorate %image NonUniform",
                "OpDecorate %result NonUniform",
                "%dst_heap_u32 = OpTypeInt 32 0",
                "%dst_heap_image_index = OpLoad %dst_heap_u32 {image_id}",
                "OpDecorate %dst_heap_image_index NonUniform",
                "%dst_heap_image_array = OpTypeRuntimeArray typeof*{image_type}",
                "%dst_heap_image_handles_ptr_type = OpTypePointer Generic %dst_heap_image_array",
                "%dst_heap_image_ptr_type = OpTypePointer Generic typeof*{image_type}",
                "%dst_heap_image_handles = OpVariable %dst_heap_image_handles_ptr_type UniformConstant",
                "%dst_heap_image_ptr = OpAccessChain %dst_heap_image_ptr_type %dst_heap_image_handles %dst_heap_image_index",
                "%image = OpLoad typeof*{image_type} %dst_heap_image_ptr",
                "%coordinate = OpLoad _ {coordinate}",
                "%result = OpImageRead typeof*{result} %image %coordinate",
                "OpStore {result} %result",
                image_id = in(reg) &self.id,
                image_type = in(reg) image_type.as_mut_ptr(),
                coordinate = in(reg) &coordinate,
                result = in(reg) &mut result,
                set = const RESOURCE_DESCRIPTOR_SET,
                binding = const RESOURCE_BINDING,
            );
            result.truncate_into()
        }
    }

    /// Writes one texel to a bindless image with unknown sampledness.
    #[inline(always)]
    pub unsafe fn write<I>(
        &self,
        coordinate: impl ImageCoordinate<I, DIM, ARRAYED>,
        texels: SampledType::SampleResult,
    ) where
        I: Integer,
    {
        #[cfg(not(target_arch = "spirv"))]
        {
            let _ = (coordinate, texels);
            unimplemented!("`BindlessImage::write` is only available on SPIR-V targets")
        }

        #[cfg(target_arch = "spirv")]
        unsafe {
            let mut image_type = MaybeUninit::<
                Image<
                    SampledType,
                    DIM,
                    DEPTH,
                    ARRAYED,
                    MULTISAMPLED,
                    { Sampled::Unknown as u32 },
                    FORMAT,
                    COMPONENTS,
                >,
            >::uninit();
            asm!(
                "OpCapability RuntimeDescriptorArray",
                "OpCapability StorageImageWriteWithoutFormat",
                "OpExtension \"SPV_EXT_descriptor_indexing\"",
                "OpDecorate %dst_heap_image_handles Binding {binding}",
                "OpDecorate %dst_heap_image_handles DescriptorSet {set}",
                "OpDecorate %image NonUniform",
                "%dst_heap_u32 = OpTypeInt 32 0",
                "%dst_heap_image_index = OpLoad %dst_heap_u32 {image_id}",
                "OpDecorate %dst_heap_image_index NonUniform",
                "%dst_heap_image_array = OpTypeRuntimeArray typeof*{image_type}",
                "%dst_heap_image_handles_ptr_type = OpTypePointer Generic %dst_heap_image_array",
                "%dst_heap_image_ptr_type = OpTypePointer Generic typeof*{image_type}",
                "%dst_heap_image_handles = OpVariable %dst_heap_image_handles_ptr_type UniformConstant",
                "%dst_heap_image_ptr = OpAccessChain %dst_heap_image_ptr_type %dst_heap_image_handles %dst_heap_image_index",
                "%image = OpLoad typeof*{image_type} %dst_heap_image_ptr",
                "%coordinate = OpLoad _ {coordinate}",
                "%texels = OpLoad _ {texels}",
                "OpImageWrite %image %coordinate %texels",
                image_id = in(reg) &self.id,
                image_type = in(reg) image_type.as_mut_ptr(),
                coordinate = in(reg) &coordinate,
                texels = in(reg) &texels,
                set = const RESOURCE_DESCRIPTOR_SET,
                binding = const RESOURCE_BINDING,
            );
        }
    }
}

/// Loads one element from a storage buffer descriptor in set `0`, binding `0`.
///
/// Prefer [`storage_buffer_from_u32`] when loading or storing more than one
/// element; this compatibility helper resolves the descriptor on every call.
#[inline(always)]
pub unsafe fn storage_buffer_load_from_u32<T: Copy + 'static>(id: u32, index: u32) -> T {
    storage_buffer_from_u32::<T>(id)[index as usize]
}

/// Stores one element into a storage buffer descriptor in set `0`, binding `0`.
///
/// Prefer [`storage_buffer_from_u32`] when loading or storing more than one
/// element; this compatibility helper resolves the descriptor on every call.
#[inline(always)]
pub unsafe fn storage_buffer_store_from_u32<T: Copy + 'static>(id: u32, index: u32, value: T) {
    storage_buffer_from_u32::<T>(id)[index as usize] = value;
}

#[cfg(target_arch = "spirv")]
#[inline(always)]
unsafe fn storage_buffer_from_u32_inner<T, P>(id: u32, value_type: *mut T, result: *mut P) {
    unsafe {
        asm!(
            "OpCapability RuntimeDescriptorArray",
            "OpExtension \"SPV_EXT_descriptor_indexing\"",
            "OpExtension \"SPV_KHR_storage_buffer_storage_class\"",
            "OpDecorate %dst_heap_resource_handles Binding {binding}",
            "OpDecorate %dst_heap_resource_handles DescriptorSet {set}",
            "OpDecorate %dst_heap_value_runtime_array ArrayStride {stride}",
            "%dst_heap_u32 = OpTypeInt 32 0",
            "%dst_heap_index = OpLoad %dst_heap_u32 {id}",
            "OpDecorate %dst_heap_index NonUniform",
            "%dst_heap_value_runtime_array = OpTypeRuntimeArray typeof*{value_type}",
            "%dst_heap_storage_buffer = OpTypeStruct %dst_heap_value_runtime_array",
            "%dst_heap_storage_buffer_array = OpTypeRuntimeArray %dst_heap_storage_buffer",
            "%dst_heap_resource_handles_ptr_type = OpTypePointer Generic %dst_heap_storage_buffer_array",
            "%dst_heap_resource_handles = OpVariable %dst_heap_resource_handles_ptr_type StorageBuffer",
            "%dst_heap_buffer_ptr = OpAccessChain typeof*{result} %dst_heap_resource_handles %dst_heap_index",
            "OpStore {result} %dst_heap_buffer_ptr",
            id = in(reg) &id,
            value_type = in(reg) value_type,
            result = in(reg) result,
            set = const RESOURCE_DESCRIPTOR_SET,
            binding = const RESOURCE_BINDING,
            stride = const core::mem::size_of::<T>(),
        );
    }
}
