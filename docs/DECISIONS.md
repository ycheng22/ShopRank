# Architecture & Technical Decisions

This document records key technical decisions, environment configurations, and dependency trade-offs.

## Local Hardware & Deep Learning Runtime Environment

- **GPU Model**: NVIDIA GeForce RTX 2060 SUPER
- **VRAM**: 8192 MiB (8 GB)
- **Driver Version**: 595.95
- **CUDA Version**: CUDA 12.4 (Driver supports up to CUDA 13.2)
- **PyTorch Version**: 2.6.0+cu124
- **vLLM Version**: N/A (Official wheels do not support native Windows; recommended to use WSL2 or local serving via Ollama/llama.cpp)


## Model providers
- OpenAI API 
- Gemeni 
- DeepSeek 
- Alibaba Model 

## Batch size decision
```json
{
  "date": "2026-08-30",
  "device": "NVIDIA GeForce RTX 2060 SUPER",
  "max_len": 512,
  "total_products": 50000,
  "results": [
    {
      "batch_size": 32,
      "items_per_s": 88.7,
      "peak_gib": 1.74
    },
    {
      "batch_size": 64,
      "items_per_s": 71.0,
      "peak_gib": 2.24
    },
    {
      "batch_size": 128,
      "items_per_s": 70.4,
      "peak_gib": 3.33
    },
    {
      "batch_size": 256,
      "items_per_s": 69.6,
      "peak_gib": 5.58
    },
    {
      "batch_size": 512,
      "items_per_s": 13.7,
      "peak_gib": 10.1
    }
  ],
  "chosen": 32,
  "note": "Throughput reached its plateau (knee of the curve). Chosen value naturally leaves a memory safety buffer."
}
```

## Cloud region decision
- All cloud resources should prioritize us-east region.
- In case of resource shortage, use us-west region.

## Local deployment vs Docker deployment
Local deployment is preferred for development and testing, while Docker deployment is preferred for production.

## GCP artifactory registry path
- us-east1-docker.pkg.dev/gcp-share-507118/containers

## Split search into cacheable 
| 2026-09-02 | Split search into cacheable `GET` (preset examples) and `POST` (free-form, BYO key); CORS allow-list from Settings, never `*` | Frontend on Cloudflare Pages and API on Cloud Run are different origins. A cross-origin POST always triggers a preflight, doubling latency on a cold start — which is exactly the first impression a recruiter gets. GET keeps preset queries CORS-simple and CDN-cacheable, so they render even while the service is cold or down. | Applies to SPEC 22.5 |