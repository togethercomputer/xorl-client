# XoRL Limitations

This document describes the known limitations of XoRL compared to Tinker.

## Single-Model Training

XoRL is designed for single-model training. Unlike Tinker which supports multiple concurrent training runs, XoRL typically operates with only the "default" training run.

**Implications:**
- The `list_training_runs()` API will usually return only one training run
- Additional training runs can be created via `/api/v1/create_model`, but they share the same underlying model weights
- The `model_id` parameter in most APIs defaults to "default"

## Local Deployment

XoRL is intended for local or self-hosted deployments, not cloud-based multi-tenant environments.

## API Compatibility

While XoRL aims for API compatibility with Tinker, some features may behave differently or have reduced functionality due to the single-model architecture.
