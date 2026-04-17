//! Caduceus Bridge — direct Rust integration of the Caduceus AI engine into Zed.
//!
//! No MCP, no JSON-RPC, no serialization overhead. Native Rust types.
//!
//! This crate provides:
//! - `CaduceusEngine` — singleton holding all engine state
//! - Tool execution via the `ToolRegistry`
//! - Semantic search via `SemanticIndex`
//! - Code intelligence via `CodePropertyGraph`
//! - Git operations via `GitRepo`
//! - Security scanning via `SastScanner`
//! - Memory persistence via filesystem
//! - Session management via `SqliteStorage`

pub mod engine;
pub mod tools;
pub mod search;
pub mod git;
pub mod security;
pub mod memory;
pub mod storage;
pub mod telemetry;
pub mod crdt;
pub mod orchestrator;
pub mod marketplace;
pub mod providers;
pub mod runtime;
pub mod safety;

pub use engine::CaduceusEngine;
