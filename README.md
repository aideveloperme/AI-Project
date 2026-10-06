# AI & Computer Vision Projects

This repository contains multiple AI and computer vision projects implemented in Python using Google Colab, focusing on deep learning, CNNs, and NLP. Each notebook is self-contained and demonstrates different AI/ML techniques applied to real-world tasks.

vllm-gb10-profiling/ — LLM serving performance on NVIDIA DGX Spark (GB10)
Description: Serve Qwen3-8B with vLLM, load-test it, profile it, and measure before/after for prefix caching, FP8 quantisation, batching and context limits, plus a tool-calling agent loop for agent latency.
Tech Stack: vLLM, Docker, Python (asyncio/httpx), Prometheus metrics, PyTorch profiler, Nsight Systems, GitHub Actions
Highlights: KV-cache and context-length analysis, goodput under SLOs, per-step agent TTFT, Perfetto profile links, CI on a GB10 simulator. See vllm-gb10-profiling/README.md.

Finding_laneLines.ipynb
Description: Lane detection from road images/videos using computer vision techniques.
Tech Stack: OpenCV, Python, NumPy, Matplotlib
Highlights: Demonstrates edge detection, Hough transforms, and lane-line tracking.

ICH_CNN_(2).ipynb
Description: Intracerebral Hemorrhage detection using Convolutional Neural Networks (CNNs).
Tech Stack: TensorFlow,Keras, Python, CNNs
Highlights: Binary class classification with model evaluation metrics (accuracy, loss curves, confusion matrix).

News_Classification.ipynb
Description: Text classification of news articles using NLP techniques.
Tech Stack: Python, Pandas, Scikit-learn, NLP preprocessing
Highlights: Implements tokenization, vectorization, and classification models ( Naive Bayes, SVM).

trafficsign_classification.ipynb
Description: Classification of traffic signs using CNN models.
Tech Stack: Python, TensorFlow, Keras, CNNs
Highlights: Image preprocessing, model training, and evaluation with high accuracy on test dataset.

How to Run
Open the notebooks in Google Colab.
Install required dependencies if prompted (TensorFlow, Keras, OpenCV, NumPy, Pandas, Scikit-learn).
Run each notebook cell by cell to reproduce the results.

Features & Learning Outcomes
Hands-on experience with CNN architectures for image classification.
NLP techniques for text preprocessing and classification.
Data visualization and performance evaluation of models.
Understanding preprocessing pipelines for images and text.
