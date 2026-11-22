"""
Presets & Benchmarks for Coordinate & Index-Based Sparse Adapter Compression:
1. CoordinateTuple: Normalized (u, v, delta_w) coordinate entry
2. ACRONYM_REFERENCE_TABLE: 3-column Acronym Reference Sheet (Acronym | Subject | Full Meaning)
3. SLIDE_AND_LLM_BENCHMARKS: 10x10 SIT Slide Matrix vs. 768x768 BERT vs. 4096x4096 LLM
4. QUANTUM_HILBERT_BENCHMARKS: Why Coordinate Compression beats LoRA in 2^n x 2^n Quantum State Spaces
"""

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class CoordinateTuple:
    u_norm: float  # Normalized row coordinate in [0.0, 1.0]
    v_norm: float  # Normalized col coordinate in [0.0, 1.0]
    delta_weight: float
    semantic_role: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompressionBenchmarkRow:
    method_name: str
    slide_10x10_params: int
    bert_768x768_params: int
    llm_4096x4096_params: int
    quantum_17qubit_hilbert_params: int
    cross_dimension_portable: bool
    surgical_zero_drift: bool


ACRONYM_REFERENCE_TABLE: List[Dict[str, str]] = [
    {"acronym": "CISAC", "subject": "Frontier PEFT (This Repo)", "full_meaning": "Coordinate & Index-Based Sparse Adapter Compression — Storing weight updates as normalized (u, v, delta_w) coordinates that project across any model dimension or quantum circuit."},
    {"acronym": "PEFT", "subject": "Model Fine-Tuning (ICT3507C)", "full_meaning": "Parameter-Efficient Fine-Tuning — Freezing 99%+ of base model weights and training only a compact plug-in adapter."},
    {"acronym": "LoRA", "subject": "Model Fine-Tuning (ICT3507C)", "full_meaning": "Low-Rank Adaptation — Factoring a 10x10 (100-weight) update matrix into a 10xr column matrix and an rx10 row matrix (20 weights at Rank 1; 40 at Rank 2)."},
    {"acronym": "QLoRA", "subject": "Model Fine-Tuning (ICT3507C)", "full_meaning": "Quantized Low-Rank Adaptation — Compressing the frozen base model to 4-bit precision while training a LoRA adapter on top."},
    {"acronym": "VeRA", "subject": "Coordinate / Index Fine-Tuning", "full_meaning": "Vector-based Random Matrix Adaptation — Freezing projection matrices and training only tiny 1D coordinate scaling vectors."},
    {"acronym": "LoRA-XS", "subject": "Coordinate / Core Fine-Tuning", "full_meaning": "Low-Rank Adaptation Extremely Small — Training only a tiny r x r core coordinate matrix (e.g., 2x2 = 4 weights instead of 40) between frozen projections."},
    {"acronym": "SFT", "subject": "LLM Training Pipeline", "full_meaning": "Supervised Fine-Tuning — Teaching a base model task-specific behavior or stripped connective syntax using curated (Input, Output) pairs."},
    {"acronym": "RLHF", "subject": "LLM Alignment", "full_meaning": "Reinforcement Learning from Human Feedback — Aligning model output probabilities using human preference rankings."},
    {"acronym": "DPO", "subject": "LLM Alignment", "full_meaning": "Direct Preference Optimization — Aligning a model directly on (Preferred, Rejected) pairs without a separate reward model."},
    {"acronym": "BPE", "subject": "Tokenization", "full_meaning": "Byte-Pair Encoding — Subword tokenization algorithm that iteratively merges frequent character pairs."},
    {"acronym": "BoW", "subject": "Text Classification", "full_meaning": "Bag-of-Words — Counting words in isolation while ignoring sentence order and contextual relationships."},
    {"acronym": "AR", "subject": "Generative Architectures", "full_meaning": "Autoregressive Model — Looks a little behind and steps forward sequentially: P(x_t | x_<t)."},
    {"acronym": "Seq2Seq", "subject": "Generative Architectures", "full_meaning": "Sequence-to-Sequence (Encoder-Decoder) — Looks both ways: encodes the full input before decoding a high-confidence output (T5 / FLAN-T5)."},
    {"acronym": "DDPM / DiT", "subject": "Diffusion Models", "full_meaning": "Denoising Diffusion Probabilistic Model / Diffusion Transformer — Refines Gaussian noise into coherent images or video frames."},
    {"acronym": "QRAM", "subject": "Quantum Machine Learning", "full_meaning": "Quantum Random Access Memory — Quantum addressing mechanism that loads sparse coordinate-indexed weights |i_k, j_k> into superposition in O(K log n) depth."},
    {"acronym": "VQE / QNN", "subject": "Quantum Machine Learning", "full_meaning": "Variational Quantum Eigensolver / Quantum Neural Network — Parameterized quantum circuits whose rotation angles act as trainable coordinate weights."},
    {"acronym": "RUS", "subject": "Trapped-Ion Quantum Control", "full_meaning": "Repeat-Until-Success — Mid-circuit ancilla measurement and adaptive phase-kick loop on trapped-ion hardware (Quantinuum Guppy)."},
]


SLIDE_AND_LLM_BENCHMARKS: List[CompressionBenchmarkRow] = [
    CompressionBenchmarkRow(
        method_name="Full Fine-Tuning (Dense d x d Matrix)",
        slide_10x10_params=100,
        bert_768x768_params=589_824,
        llm_4096x4096_params=16_777_216,
        quantum_17qubit_hilbert_params=17_179_869_184,
        cross_dimension_portable=False,
        surgical_zero_drift=False,
    ),
    CompressionBenchmarkRow(
        method_name="Standard LoRA (Rank r = 2 on SIT Slide)",
        slide_10x10_params=40,
        bert_768x768_params=3_072,
        llm_4096x4096_params=16_384,
        quantum_17qubit_hilbert_params=524_288,
        cross_dimension_portable=False,
        surgical_zero_drift=False,
    ),
    CompressionBenchmarkRow(
        method_name="Standard LoRA (Rank r = 1 on SIT Slide)",
        slide_10x10_params=20,
        bert_768x768_params=1_536,
        llm_4096x4096_params=8_192,
        quantum_17qubit_hilbert_params=262_144,
        cross_dimension_portable=False,
        surgical_zero_drift=False,
    ),
    CompressionBenchmarkRow(
        method_name="Coordinate / Index-Based Sparse Adapter (K = 6 Active Coordinates)",
        slide_10x10_params=6,
        bert_768x768_params=6,
        llm_4096x4096_params=6,
        quantum_17qubit_hilbert_params=6,
        cross_dimension_portable=True,
        surgical_zero_drift=True,
    ),
]


QUANTUM_HILBERT_BENCHMARKS: List[Dict[str, str]] = [
    {
        "register_size": "4 Qubits (16 x 16 Hilbert Matrix)",
        "full_unitary_params": "256",
        "rank1_lora_params": "32",
        "coordinate_adapter_k6_gates": "6 Native Gates (PhasedX / ZZPhase)",
        "quantum_advantage_note": "1-to-1 compilation of (i_k, j_k, theta_k) coordinates to Pauli rotation gates.",
    },
    {
        "register_size": "9 Qubits (512 x 512 Q-Rotate Default Register)",
        "full_unitary_params": "262,144",
        "rank1_lora_params": "1,024",
        "coordinate_adapter_k6_gates": "6 to 16 Native Gates (0 SWAPs, Depth < 20)",
        "quantum_advantage_note": "Avoids Barren Plateaus by restricting variational gradient flow to K high-sensitivity Lie coordinates.",
    },
    {
        "register_size": "17 Qubits (131,072 x 131,072 High-Res Register)",
        "full_unitary_params": "17,179,869,184",
        "rank1_lora_params": "262,144 (Uncompilable before decoherence!)",
        "coordinate_adapter_k6_gates": "6 to 32 Native Gates (QRAM O(K log n) loading)",
        "quantum_advantage_note": "Rank-1 LoRA still scales as O(2^n) and exceeds coherence time; Coordinate Compression stays O(K)!",
    },
]


DEFAULT_COORDINATE_ADAPTER_PACK: List[CoordinateTuple] = [
    CoordinateTuple(0.10, 0.20, +0.85, "Domain Ontology Anchor (Greek Myth / Anime / LARP Routing)"),
    CoordinateTuple(0.30, 0.70, -0.64, "Connective-Word Suppression Gate (Strips Low-Entropy Filler)"),
    CoordinateTuple(0.50, 0.50, +0.92, "Audience & Geopolitical Lens Pivot (Straits Times vs. Bloomberg)"),
    CoordinateTuple(0.70, 0.30, +0.78, "Autoregressive World-State Lock (Church Door #5E1914 & Brass Handle)"),
    CoordinateTuple(0.80, 0.90, -0.71, "Cognitive-Warfare Anomaly Gate (Flags Coercive Urgency Manipulation)"),
    CoordinateTuple(0.90, 0.10, +0.88, "Quantum SU(2) Phase-Kick Coordinate (Maps to Trapped-Ion ZZPhase)"),
]
