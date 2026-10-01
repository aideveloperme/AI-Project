# AI & Computer Vision Projects

This repository contains multiple AI and computer vision projects implemented in Python using Google Colab, focusing on deep learning, CNNs, and NLP. Each notebook is self-contained and demonstrates different AI/ML techniques applied to real-world tasks.

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

spark-edge-profiling/
Description: End-to-end inference profiling & optimisation study on NVIDIA DGX Spark (GB10): FP32 / FP16 / INT8 PTQ / INT8 QAT / mixed precision, Conv+BN+ReLU fusion with per-layer timing, NCHW vs NHWC and tiling, DRAM-traffic measurement with Nsight Compute, model surgery (SiLU→ReLU6, resolution, ViT head pruning) and a roofline-based report.
Tech Stack: PyTorch, TensorRT, NVIDIA ModelOpt, Triton, Nsight Compute, NVML
How to run: see spark-edge-profiling/README.md (./docker/run.sh then ./run_all.sh)

How to Run
Open the notebooks in Google Colab.
Install required dependencies if prompted (TensorFlow, Keras, OpenCV, NumPy, Pandas, Scikit-learn).
Run each notebook cell by cell to reproduce the results.

Features & Learning Outcomes
Hands-on experience with CNN architectures for image classification.
NLP techniques for text preprocessing and classification.
Data visualization and performance evaluation of models.
Understanding preprocessing pipelines for images and text.
