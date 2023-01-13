// main source file of fuga-json-seq-parser crate
// Copyright 2023 Kenta Ida 
// SPDX-License-Identifier: MIT
//

//! # fuga-json-seq-parser crate
//!
//! `fuga-json-seq-parser` is a a JSON Parser which parses JSON data sequentially, and do not generate any large deserialized data structure while parsing.

#![no_std]
mod json;

pub use json::*;