# Architecture

WeaveVision is a local modular monolith. CLI and Streamlit are delivery adapters. Application
services orchestrate domain contracts, dataset governance, framework-hidden model adapters,
inference, evaluation, reporting, registry, and SQLite persistence. Data, model artifacts,
threshold artifacts, and experiment evidence remain separate and hash-addressed.
