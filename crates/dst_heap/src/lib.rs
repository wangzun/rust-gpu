//! Descriptor-indexing helpers for rust-gpu shaders.
//!
//! This crate provides the small shader-side pieces Dust/Pumicite needs for a
//! bindless storage-buffer table implemented with `SPV_EXT_descriptor_indexing`.
//! The host side owns descriptor set `0`, binding `0`; shader code can load and
//! store typed elements from that runtime descriptor array by passing a `u32`
//! descriptor index and element index.
//!
//! Basic usage:
//!
//! ```ignore
//! use dst_heap::shaders::{storage_buffer_load_from_u32, storage_buffer_store_from_u32};
//!
//! #[spirv(compute(threads(8, 8, 1)))]
//! pub fn main_cs(src: u32, dst: u32) {
//!     let value = unsafe { storage_buffer_load_from_u32::<f32>(src, 0) };
//!     unsafe { storage_buffer_store_from_u32(dst, 0, value) };
//! }
//! ```

#![no_std]
#![cfg_attr(target_arch = "spirv", feature(asm_experimental_arch))]

pub mod shaders;

pub use shaders::*;
