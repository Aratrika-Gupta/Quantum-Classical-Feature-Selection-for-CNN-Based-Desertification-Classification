Quantum-Classical Feature Selection for CNN-Based Desertification Classification: A QAOA Extension

Extension to: Womanium AI+Climate Project (Selden, Gupta, Elgendy, Gupta, 2024)

This project extends the original Womanium AI+Climate desertification monitoring pipeline. While the original work identified pixel-level Quantum Image Processing (QIP) as infeasible due to extreme qubit requirements, this extension demonstrates a tractable hybrid quantum-classical alternative: QAOA-based feature selection on dense latent embeddings.

Instead of processing raw pixels on quantum hardware, we extract the 128-dimensional embedding from a trained Convolutional Neural Network (CNN) and formulate the feature selection process as a Quadratic Unconstrained Binary Optimization (QUBO) problem. The Quantum Approximate Optimization Algorithm (QAOA) is then used to find a compact, non-redundant subset of features that maximizes classification performance.

Key Highlights

Bypasses QIP Bottleneck: Maps feature selection (1 qubit per candidate feature) rather than raw images (1 qubit per pixel) to quantum circuits.

Balanced Objective: Optimizes a QUBO energy landscape balancing feature relevance (Mutual Information), redundancy suppression (Pearson correlation), and a soft cardinality constraint ($k$ features).

Leakage-Safe Protocol: Rigorously isolates training, validation, and test splits across CNN retraining, QUBO parameter estimation, and downstream evaluation.

Empirical Benchmarks: Achieves a mean AUC of 0.967 and mean accuracy of 0.857 across repeated experimental runs using only $k=4$ features out of a 10-candidate pool, outperforming classical Mutual Information ranking in both average score and run-to-run stability.

Pipeline Overview

 ┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
 │  Satellite Image│ ──► │ Trained CNN Backbone │ ──► │ 128-D Feature Vector │
 └─────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                 │
 ┌─────────────────┐     ┌──────────────────────┐     ┌──────────▼───────────┐
 │ Selected Subset │ ◄── │    QAOA / QUBO       │ ◄── │ MI Pre-filtering     │
 │ (k = 4 Features)│     │  (Qiskit Simulator)  │     │ (Top N Shortlist)    │
 └────────┬────────┘     └──────────────────────┘     └──────────────────────┘
          │
 ┌────────▼────────┐
 │  Random Forest  │ ──► [ Cloudy | Desert | Green Area | Water ]
 │   Classifier    │
 └─────────────────┘


Feature Extraction: A 4-block CNN trained on satellite imagery is truncated at its penultimate dense layer, generating a 128-dimensional embedding vector per image.

Classical Pre-Filtering: Mutual Information (MI) between each embedding dimension and the ground-truth target is computed on the training split to select a candidate shortlist of $N$ dimensions.

QUBO Formulation & QAOA Solve: Candidate dimensions are mapped to an Ising Hamiltonian and optimized via QAOA using a classical optimizer (COBYLA) on Qiskit's statevector simulator.

Downstream Evaluation: A Random Forest classifier (200 trees) is trained on the QAOA-selected features and evaluated on an untouched test split.

QUBO Mathematical Formulation

Let $x_i \in \{0, 1\}$ be a binary decision variable indicating whether shortlisted embedding dimension $i$ is selected. The objective function minimizes energy $H(x)$:

$$H(x) = -\alpha \sum_{i} \text{relevance}_i \cdot x_i + \beta \sum_{i < j} \text{redundancy}_{ij} \cdot x_i x_j + \gamma \left( \sum_{i} x_i - k \right)^2$$

Where:

$\text{relevance}_i$: Min-max normalized Mutual Information between feature $i$ and class labels.

$\text{redundancy}_{ij}$: Absolute Pearson correlation coefficient between features $i$ and $j$.

$\left(\sum_{i} x_i - k\right)^2$: Soft cardinality penalty enforcing exactly $k$ selected features.

$\alpha, \beta, \gamma$: Hyperparameter weights balancing relevance, redundancy, and constraint enforcement.

Experimental Setup & Parameters

Dataset Split: 3,941 training / 563 validation / 1,127 test images.

Classes: Cloudy, Desert, Green Area, Water.

CNN Architecture: 4 convolutional blocks, 128-unit dense embedding layer, Adam optimizer, sparse categorical cross-entropy loss (trained for 5 epochs).

Quantum Backend: Qiskit Optimization & Algorithms (statevector simulator, COBYLA optimizer, circuit depth $\text{reps} = 1$).

Evaluation Model: Downstream Random Forest Classifier (200 estimators).

Experimental Results

Experiments were conducted on two primary setups: a preliminary proof-of-concept run ($N=6, k=3$) and a repeated main benchmark ($N=10, k=4$).

Main Benchmark ($N = 10 \text{ shortlist}, k = 4 \text{ target features}$)

Across two fully repeated runs (including CNN retraining from scratch):

Method

Metric

Run 1

Run 2

Mean

Range

QAOA-Selected

AUC

0.972

0.961

0.967

0.961 – 0.972



Accuracy

0.867

0.846

0.857

0.846 – 0.867

Classical Top-$K$ (MI Only)

AUC

0.919

0.958

0.939

0.919 – 0.958



Accuracy

0.773

0.862

0.818

0.773 – 0.862

Full Embedding (Baseline)

AUC

0.992

0.993

0.993

0.992 – 0.993

(128 Dimensions)

Accuracy

0.931

0.935

0.933

0.931 – 0.935

Result Highlights & Insights

Performance & Stability: QAOA features achieved higher mean performance (+2.8 AUC points, +3.9 accuracy points over classical ranking) with lower run-to-run variance (accuracy spread of 0.021 vs 0.089 for classical top-$K$).

Optimization Convergence: QAOA consistently converged to a stable minimum energy (approximately $-32.25$) in both runs.

Variance Source: Variance between runs is largely driven by stochasticity in retraining the upstream CNN backbone rather than instability in the QAOA solver itself.

Wall-Clock Scaling: Increasing candidate size from $N=6$ to $N=10$ increased QAOA simulation time from ~2 seconds to ~678 seconds, highlighting classic statevector simulation scaling limits.

Data Leakage Prevention Protocol

To ensure true generalizability, the evaluation pipeline strictly enforces the following separation:

Data partitioning into Train / Val / Test occurs before any network training or analysis.

The CNN is retrained from scratch using only the training split.

Mutual Information scores and Pearson correlation matrices are calculated exclusively on training-set embeddings.

Downstream Random Forest classifiers are fit on training-set feature subsets and evaluated once on the untouched test split.

Repository Structure

.
├── models/
│   └── Model.h5                 # Trained CNN backbone
├── src/
│   ├── feature_extraction.py    # Extracts 128-D embeddings from CNN
│   ├── qubo_builder.py          # Builds relevance/redundancy matrices & QUBO
│   ├── qaoa_solver.py           # Qiskit QAOA execution & COBYLA optimizer
│   └── evaluate.py              # Random forest evaluator & leakage-safe pipeline
├── main.py                      # Main entry point for end-to-end execution
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


Future Roadmap

[ ] Decouple CNN Stochasticity: Freeze CNN backbone weights to isolate QAOA feature-selection stability from embedding variance.

[ ] Scale Candidate Size: Expand candidate shortlist to $N = 14\text{–}20$ features using gradient-free optimizers built for noisy spaces (e.g., SPSA).

[ ] Hardware Execution: Transpile and execute the feature selection QUBO on real quantum hardware (e.g., IBM Quantum QPUs).

[ ] End-to-End Fine-Tuning: Evaluate performance when directly retraining the CNN's dense classification layer on the selected feature indices.

Citation & References

If you use or reference this work, please cite the original project and this extension:

@article{selden2024womanium,
  title={Womanium AI+Climate Project: Satellite Imagery for Desertification Monitoring},
  author={Selden, M. and Gupta, P. and Elgendy, I. M. and Gupta, A.},
  year={2024}
}

@misc{farhi2014quantum,
  title={A Quantum Approximate Optimization Algorithm},
  author={Farhi, Edward and Goldstone, Jeffrey and Gutmann, Sam},
  year={2014},
  eprint={1411.4028},
  archivePrefix={arXiv},
  primaryClass={quant-ph}
}
