"""servebench: load-testing, agent-latency and profiling harness for vLLM.

Built for an NVIDIA DGX Spark (GB10) but works against any OpenAI-compatible
vLLM endpoint. A GB10-like simulator (``servebench.mock``) lets the whole
pipeline run in CI without a GPU.
"""

__version__ = "0.1.0"
