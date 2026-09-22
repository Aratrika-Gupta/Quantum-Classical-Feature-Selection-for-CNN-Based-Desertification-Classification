Quantum-Classical Feature Selection for CNN-Based Desertification
Classification: A QAOA Extension
Extension to: Womanium AI+Climate Project (Selden, Gupta, Elgendy, Gupta, 2024)

Abstract
The original Womanium AI+Climate project introduced a convolutional neural
network (CNN) for classifying satellite imagery into land-cover categories relevant
to desertification monitoring (Cloudy, Desert, Green Area, Water), reporting
near-random baseline performance and identifying Quantum Image Processing (QIP)
as currently infeasible due to prohibitive qubit requirements at the pixel level.
This extension addresses a different, tractable point of entry for quantum
computing in the same pipeline: feature selection. Rather than operating on raw
pixels, we extract the CNN's learned embedding for each image and apply the
Quantum Approximate Optimization Algorithm (QAOA) to select a small, non-redundant
subset of embedding dimensions, formulated as a Quadratic Unconstrained Binary
Optimization (QUBO) problem balancing feature relevance against redundancy. We
benchmark QAOA-selected features against a classical mutual-information filter and
the full embedding on a held-out test set, and repeat the experiment to assess
result stability. Across two repeated runs at a shortlist size of 10 candidate
features, QAOA-selected features achieved a mean AUC of 0.967 (range 0.961-0.972)
and mean accuracy of 0.857 (range 0.846-0.867), narrowly ahead of and more stable
than classical top-K selection (mean AUC 0.939, range 0.919-0.958; mean accuracy
0.818, range 0.773-0.862). We report this instability honestly: a large advantage
for QAOA observed in the first run did not fully replicate in the second, and we
discuss likely sources of variance and next steps toward a more conclusive result.

Keywords: desertification; convolutional neural networks (CNN); feature selection;
QAOA; QUBO; quantum-classical hybrid methods

1 Introduction and Motivation
The original project's CNN pipeline (Section 3, original paper) produces a
128-dimensional learned embedding per image ahead of its final classification
layer. The paper's discussion of quantum methods (Section 3.1) considered Quantum
Neural Networks and Quantum Image Processing, but discarded the latter due to the
low pixel counts current QIP methods support. This limitation is specific to
pixel-level quantum encoding; it does not apply to a combinatorial problem defined
over a small number of engineered or learned features rather than raw pixels.

Feature selection is naturally combinatorial: given N candidate features, choosing
the best subset of size K is a search over a discrete space that grows
combinatorially with N. This is precisely the class of problem QAOA is designed
for [FGG14]. We therefore reframe the CNN's embedding-dimension selection problem
as a QUBO and solve it with QAOA, using one qubit per candidate feature rather than
per pixel -- keeping the problem within reach of a classical simulator even though
QIP on raw imagery remains impractical.

2 Methodology
2.1 Feature extraction
The trained CNN (Model.h5; four convolutional blocks, a 128-unit dense embedding
layer, and a four-class softmax head, per the original architecture) is truncated
at its penultimate dense layer, yielding a 128-dimensional embedding vector per
input image.

2.2 Classical pre-filtering
Mutual information between each embedding dimension and the class label is
computed on the training split only, and the top N dimensions ("shortlist") are
retained as QAOA's candidate pool. This keeps the QUBO at a realistic qubit count
(one qubit per shortlisted dimension) rather than one per raw embedding dimension.

2.3 QUBO formulation
Let x_i in {0,1} indicate whether shortlisted dimension i is selected. We minimize:

  -alpha * sum_i(relevance_i * x_i)
  + beta  * sum_{i<j}(redundancy_ij * x_i * x_j)
  + gamma * (sum_i(x_i) - k)^2

where relevance_i is min-max normalized mutual information, redundancy_ij is the
absolute Pearson correlation between shortlisted dimensions i and j, and the third
term is a soft cardinality constraint enforcing exactly k selected features.

2.4 QAOA solver
The QUBO is converted to an Ising Hamiltonian and solved with QAOA
(qiskit-optimization, qiskit-algorithms) using COBYLA as the classical outer-loop
optimizer, run on Qiskit's statevector simulator. Circuit depth (reps) and
optimizer iteration budget (maxiter) are treated as tunable hyperparameters.

2.5 Leakage-safe evaluation protocol
A methodological concern specific to this extension: because the CNN is retrained
as part of each experimental run, care must be taken that no information from the
held-out test set leaks into feature selection or model fitting. We enforce a
strict split -- training, validation, and test images are partitioned once before
any step; the CNN is retrained from scratch on the training split only; mutual
information and QUBO weights are computed from training-split embeddings only; and
final AUC/accuracy are reported exclusively on the untouched test split. This
addresses a leakage issue identified during development, in which an
already-trained model's embeddings had implicitly memorized the evaluation images,
producing artificially high (>0.97) AUC scores under naive evaluation.

3 Experimental Setup
Each run: CNN retrained for 5 epochs (Adam optimizer, sparse categorical
cross-entropy loss, batch size 32) on a fresh train/validation/test split (3,941 /
563 / 1,127 images). A downstream Random Forest classifier (200 trees) is fit on
selected features from the training split and evaluated on the test split. Three
feature-selection methods are compared: QAOA-selected, classical top-K (mutual
information ranking alone, no redundancy term), and the full 128-dimensional
embedding as an upper-bound reference.

4 Results
4.1 Preliminary validation (shortlist = 6, k = 3)
An initial, reduced-scale run confirmed the pipeline executes correctly end to end
and that QAOA converges to a stable objective value. At this small shortlist size,
QAOA-selected and classical top-K features performed comparably (AUC 0.982 vs.
0.983; accuracy 0.906 vs. 0.890), which is expected: with only 3 of 6 candidates to
choose, there is limited room for a redundancy-aware method to diverge from a
relevance-only ranking.

4.2 Main experiment (shortlist = 10, k = 4), repeated twice

  Run 1:
    QAOA-selected              AUC 0.972   Accuracy 0.867
    Classical top-K (MI only)  AUC 0.919   Accuracy 0.773
    Full embedding (baseline)  AUC 0.992   Accuracy 0.931

  Run 2:
    QAOA-selected              AUC 0.961   Accuracy 0.846
    Classical top-K (MI only)  AUC 0.958   Accuracy 0.862
    Full embedding (baseline)  AUC 0.993   Accuracy 0.935

  Mean across runs (range in parentheses):
    QAOA-selected              AUC 0.967 (0.961-0.972)   Accuracy 0.857 (0.846-0.867)
    Classical top-K            AUC 0.939 (0.919-0.958)   Accuracy 0.818 (0.773-0.862)
    Full embedding             AUC 0.993 (0.992-0.993)   Accuracy 0.933 (0.931-0.935)

QAOA converged to a stable objective value in both runs (approximately -32.25),
indicating the optimizer itself is reliably finding the QUBO's minimum; the
variability observed instead concentrates in the downstream classification metrics.

5 Discussion
Run 1 showed a substantial QAOA advantage over classical top-K selection (+5.3 AUC
points, +9.4 accuracy points). Run 2 largely erased this gap, and classical top-K
even edged ahead on accuracy. We report both runs rather than the more favorable
one: the correct interpretation at this stage is that QAOA-selected features show
a modest average AUC/accuracy advantage and, notably, substantially lower run-to-run
variance (accuracy range of 0.021 versus 0.089 for classical top-K), but a firm
claim that QAOA outperforms classical selection is not yet supported by two runs
alone.

A likely source of the observed variance is that the CNN is retrained from scratch
in each run, so the 128-dimensional embedding itself differs between runs -- not
only the stochastic elements of QAOA (COBYLA's optimization path) or classical
selection. This confounds "instability of the selection method" with "instability
of the underlying embedding it selects from." Isolating these two sources of
variance (e.g., by fixing the CNN's weights and only re-running feature selection)
is identified as a priority for follow-up work.

Separately, QAOA's wall-clock cost scaled non-linearly with qubit count in our
runs: increasing the shortlist from 6 to 10 candidates (with a five-fold increase
in optimizer iterations) increased QAOA runtime from approximately 2 seconds to
678 seconds, a larger increase than qubit count and iteration budget alone would
predict, likely reflecting sparse linear-algebra overhead in the simulator at
higher qubit counts.

6 Limitations
- All experiments use small qubit counts (6-10) and shallow circuits (reps = 1) on
  a classical simulator; results should be read as a proof of concept for the QUBO
  formulation rather than a demonstration of quantum advantage.
- Only two repeated runs have been completed at the main experimental setting;
  the observed instability warrants additional repeats before drawing firm
  conclusions.
- CNN retraining stochasticity is not currently separated from feature-selection
  stochasticity (see Discussion).
- A Random Forest is used as the evaluation classifier for simplicity; results may
  differ if the CNN's own classification head were retrained on selected features
  directly.

7 Conclusion and Future Work
This extension demonstrates that QAOA-based feature selection is a technically
feasible and methodologically sound addition to the original CNN pipeline,
avoiding the qubit-count barrier that ruled out pixel-level quantum image
processing in the original paper. Held-out evaluation confirms the approach is
free of data leakage, and QAOA reliably converges to a stable QUBO solution.
Whether QAOA-selected features meaningfully outperform classical mutual-information
selection remains an open question pending further repeated runs; the current
evidence favors QAOA on stability (lower run-to-run variance) more clearly than on
raw average performance. Planned next steps: (i) additional repeated runs to
establish confidence intervals, (ii) decoupling CNN-retraining variance from
feature-selection variance, (iii) scaling the candidate shortlist to 14-20
dimensions with a more runtime-efficient classical optimizer (e.g. SPSA), and
(iv) evaluating the selected QUBO on real quantum hardware.

References
[FGG14] E. Farhi, J. Goldstone, S. Gutmann. A Quantum Approximate Optimization
Algorithm. arXiv:1411.4028, 2014.

[Selden24] M. Selden, P. Gupta, I. M. Elgendy, A. Gupta. Womanium AI+Climate
Project. 2024. (Original paper this work extends.)

Additional references for the CNN baseline, dataset, and case study area are as
listed in the original paper's References section.
