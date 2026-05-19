//! Descriptor-indexing helpers for rust-gpu shaders.
//!
//! This crate provides the small shader-side pieces Dust/Pumicite needs for a
//! bindless storage-buffer table implemented with `SPV_EXT_descriptor_indexing`.
//! The host side owns descriptor set `0`, binding `0` for resources and
//! descriptor set `1`, binding `0` for samplers.
//!
//! Basic usage:
//!
//! ```ignore
//! use dst_heap::shaders::storage_buffer_from_u32;
//!
//! #[spirv(compute(threads(8, 8, 1)))]
//! pub fn main_cs(src: u32, dst: u32) {
//!     let values = storage_buffer_from_u32::<f32>(src);
//!     values[1] = values[0];
//! }
//! ```

#![no_std]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

pub mod shaders;

pub use shaders::*;

// Descriptor-indexing helpers implemented now:
// pub fn storage_buffer_from_u32<T>(index: u32) -> &mut TypedBuffer<[T]>;
// pub fn image_from_u32<I>(index: u32) -> BindlessImage<I>;
// pub fn sampler_from_u32(index: u32) -> BindlessSampler;
// Storage buffers resolve the descriptor once. Deref coercion gives `&mut [T]`,
// so Rust indexing can be used repeatedly without re-emitting descriptor
// declarations for every element.
