# 🧠 Deep Learning Roadmap

> *A structured, opinionated guide for mastering deep learning — assumes ML fundamentals are solid.*

---

## Stage 1 — Neural Network Fundamentals

*The bedrock. Even if you've seen some of this, make sure it's airtight.*

### Core Concepts
- [ ] Perceptrons → MLPs → Deep Networks
- [ ] Activation Functions: ReLU, Leaky ReLU, Sigmoid, Tanh, GELU, Swish
- [ ] Forward Pass & **Backpropagation** — derive it by hand at least once
- [ ] Weight Initialization: Xavier (Glorot), He (Kaiming)
- [ ] Vanishing & Exploding Gradients — causes and fixes
- [ ] Universal Approximation Theorem (intuition)

### Normalization
- [ ] Batch Normalization
- [ ] Layer Normalization
- [ ] Group Normalization
- [ ] When to use which and why

### Optimizers
| Optimizer | Key Idea |
|---|---|
| SGD + Momentum | Classical, still competitive with tuning |
| RMSProp | Adaptive learning rates per parameter |
| **Adam** ✅ | Momentum + adaptive rates — default choice |
| **AdamW** ✅ | Adam with decoupled weight decay |
| Lion, Sophia 🔬 | Recent research optimizers |

### Regularization
- [ ] L1 / L2 Weight Decay
- [ ] Dropout & Variational Dropout
- [ ] Label Smoothing
- [ ] Stochastic Depth
- [ ] Data Augmentation as regularization

### 📚 Resources
- **Book**: *Deep Learning* — Goodfellow, Bengio, Courville *(free at deeplearningbook.org)*
- **Course**: [fast.ai Practical Deep Learning](https://course.fast.ai/)
- **Course**: DeepLearning.AI Specialization (Coursera — Andrew Ng)

---

## Stage 2 — Frameworks & Training Infrastructure

*Pick one framework, go deep. Learn to train properly.*

### Framework — Pick One First

| Framework | Best for |
|---|---|
| **PyTorch** ⭐ | Research, flexibility, Pythonic feel — recommended |
| **TensorFlow / Keras** | Production pipelines, mobile/edge deployment |
| **JAX** 🔬 | High-performance research, XLA, functional style |

**PyTorch Essentials to Master:**
```python
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))  # residual connection
```

### Training at Scale
- [ ] Learning Rate Schedules: Cosine Annealing, Linear Warmup, OneCycleLR
- [ ] Gradient Clipping (`torch.nn.utils.clip_grad_norm_`)
- [ ] Mixed Precision Training: `fp16` / `bf16` with `torch.cuda.amp`
- [ ] Gradient Accumulation (simulate large batch sizes)
- [ ] Multi-GPU: `DataParallel` → `DistributedDataParallel`
- [ ] Checkpointing & resuming training

### Experiment Tracking
- **Weights & Biases (wandb)** ⭐ — industry standard
- MLflow — open source alternative
- TensorBoard — built into TF, works with PyTorch too

### 📚 Resources
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Andrej Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Weights & Biases Courses](https://www.wandb.courses/)

---

## Stage 3 — Core Architectures

*The landmark architectures you must understand inside-out.*

### 3.1 Convolutional Neural Networks (CNNs)

> **Domain**: Images, video, spatial data, audio spectrograms

**Concepts:**
- [ ] Convolution, Padding, Stride, Dilation
- [ ] Pooling: Max, Average, Global Average
- [ ] Receptive Field
- [ ] Depthwise Separable Convolutions (MobileNet)
- [ ] Skip / Residual Connections

**Architecture Progression:**
```
LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015)
     → DenseNet → EfficientNet → ConvNeXt (2022)
```

**Must-Read Papers:**
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [A ConvNet for the 2020s (ConvNeXt)](https://arxiv.org/abs/2201.03545)

---

### 3.2 Recurrent Neural Networks (RNNs)

> **Domain**: Sequences, time-series, text *(pre-Transformer era — still relevant for edge/streaming)*

- [ ] Vanilla RNN & the vanishing gradient problem
- [ ] **LSTM** — gates, cell state, hidden state
- [ ] **GRU** — simplified gating
- [ ] Bidirectional RNNs
- [ ] Sequence-to-Sequence + Attention (the bridge to Transformers)

**Key Paper:**
- [Long Short-Term Memory (LSTM)](https://www.bioinf.jku.at/publications/older/2604.pdf) — Hochreiter & Schmidhuber, 1997

---

### 3.3 ⭐ Transformers — The Dominant Architecture

> **Non-negotiable. Everything in modern DL revolves around this.**

**Concepts — build up in this order:**
```
Dot-Product Attention
       ↓
Scaled Dot-Product Attention  →  Softmax(QKᵀ / √dₖ) · V
       ↓
Multi-Head Self-Attention
       ↓
Transformer Block (Attn + FFN + LayerNorm + Residual)
       ↓
Full Encoder / Decoder / Encoder-Decoder
```

- [ ] Queries, Keys, Values — what they represent
- [ ] Positional Encoding: Sinusoidal → RoPE → ALiBi
- [ ] Causal (Masked) Self-Attention
- [ ] Cross-Attention
- [ ] KV-Cache & inference efficiency

**Architecture Variants:**
| Model | Type | Use Case |
|---|---|---|
| BERT | Encoder-only | Classification, embeddings |
| GPT series | Decoder-only | Text generation |
| T5, BART | Encoder-Decoder | Translation, summarization |
| ViT | Encoder (patches) | Vision |

**Essential Papers:**
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
2. [BERT](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
3. [GPT-3](https://arxiv.org/abs/2005.14165) — Brown et al., 2020
4. [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)

**To implement from scratch:**
- [Andrej Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) ⭐

---

### 3.4 Generative Models

> **Domain**: Image synthesis, data augmentation, representation learning

#### Variational Autoencoders (VAEs)
- [ ] Encoder → Latent Space (μ, σ) → Reparameterization → Decoder
- [ ] ELBO loss (reconstruction + KL divergence)
- [ ] Disentangled representations

**Paper**: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

---

#### Generative Adversarial Networks (GANs)
- [ ] Generator vs Discriminator min-max game
- [ ] Mode collapse, training instability
- [ ] DCGAN → StyleGAN → BigGAN

**Papers:**
- [GAN](https://arxiv.org/abs/1406.2661) — Goodfellow et al., 2014
- [StyleGAN2](https://arxiv.org/abs/1912.04958)

---

#### ⭐ Diffusion Models — Current SOTA for Generation
- [ ] Forward process: add noise step by step
- [ ] Reverse process: learn to denoise
- [ ] DDPM, DDIM (faster sampling)
- [ ] Classifier-Free Guidance (CFG)
- [ ] Latent Diffusion (Stable Diffusion)

**Papers:**
- [DDPM](https://arxiv.org/abs/2006.11239) — Ho et al., 2020
- [Improved DDPM](https://arxiv.org/abs/2102.09672)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

---

## Stage 4 — Modern Techniques

*What separates practitioners from beginners.*

### Efficient Fine-Tuning (PEFT)
- [ ] **LoRA** — Low-Rank Adaptation of large models ⭐
- [ ] QLoRA — quantized LoRA for consumer GPUs
- [ ] Prefix Tuning, Prompt Tuning
- [ ] Adapter layers

**Paper**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### Self-Supervised & Contrastive Learning
- [ ] **CLIP** — image-text contrastive learning
- [ ] **SimCLR**, MoCo — visual self-supervision
- [ ] **DINO / DINOv2** — self-distillation
- [ ] Masked Autoencoders (MAE)

### Scaling Laws & LLM Training
- [ ] Chinchilla scaling laws (compute-optimal training)
- [ ] Data quality > data quantity
- [ ] Tokenization: BPE, SentencePiece, tiktoken
- [ ] Flash Attention, Grouped Query Attention
- [ ] RLHF — Reinforcement Learning from Human Feedback

### 📚 Resources
- [Lilian Weng's Blog](https://lilianweng.github.io/) — best DL blog on the internet
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

---

## Stage 5 — Specializations

*Pick your lane(s). Go deep.*

### 🖼️ Computer Vision
- Object Detection: YOLO series, Faster R-CNN, DETR
- Segmentation: U-Net, Mask R-CNN, SAM (Segment Anything Model)
- Depth Estimation, Optical Flow
- 3D Vision: NeRF, Gaussian Splatting

### 📝 NLP / LLMs
- RAG (Retrieval-Augmented Generation)
- Agent frameworks & tool use
- Long-context models: Mamba, RWKV, Ring Attention
- Benchmarks: MMLU, HellaSwag, BIG-Bench

### 🎮 Reinforcement Learning
- Policy Gradient: REINFORCE, PPO, SAC
- Deep RL: DQN, A3C, TD3
- RLHF & Constitutional AI
- Environments: Gymnasium, MuJoCo, Atari

### 🔊 Audio & Speech
- Spectrogram, MFCC features
- Wav2Vec 2.0, HuBERT, Whisper
- Text-to-Speech: VITS, Tortoise, Bark

### 🧬 AI for Science 🔬
- Graph Neural Networks (GNNs)
- AlphaFold 2 & protein structure prediction
- Neural ODEs & PINNs

---

## Stage 6 — Research & Production

### Reading Research Papers
1. Read the abstract + conclusion first
2. Look at all figures before the text
3. Skim the intro, then method section carefully
4. Reproduce key results in code

**Where to find papers:**
- [arxiv.org](https://arxiv.org) — cs.LG, cs.CV, cs.CL
- [Papers With Code](https://paperswithcode.com) ⭐ — benchmarks + code
- [Semantic Scholar](https://www.semanticscholar.org)
- [Hugging Face Daily Papers](https://huggingface.co/papers)

### MLOps & Production
- [ ] Model serialization: ONNX, TorchScript, TensorRT
- [ ] Quantization: INT8, INT4, GPTQ, AWQ
- [ ] Serving: vLLM, TorchServe, Triton Inference Server
- [ ] Monitoring: data drift, model degradation

---

## 🗂️ Essential Practice Projects

| Project | Skills Covered |
|---|---|
| Train MNIST from scratch in NumPy | Backprop, weight init, no framework magic |
| Implement GPT-2 from scratch | Transformers, attention, generation |
| Fine-tune ResNet on custom dataset | Transfer learning, CNNs, augmentation |
| Build a text classifier with BERT | Hugging Face, fine-tuning, PEFT |
| Train a diffusion model on small dataset | Generative models, noise schedules |
| Kaggle competition (top 20%) | End-to-end pipeline, ensembling, tricks |

---

## 📖 Canonical Books

| Book | Best for |
|---|---|
| *Deep Learning* — Goodfellow et al. | Theory & foundations |
| *Dive into Deep Learning* (d2l.ai) | Code-first, free online |
| *The Little Book of Deep Learning* | Quick, dense reference |

---

## ⏱️ Suggested Timeline

```
Day 1-5      →  Stage 1 + 2  (Foundations + Frameworks)
Day 5–14     →  Stage 3      (Architectures — CNN, RNN, Transformer)
Day 14-30    →  Stage 3.4    (Generative Models)
Day 30-50    →  Stage 4      (Modern Techniques)
Day 50-70    →  Stage 5      (Specialization of your choice)
Day 70-90    →  Stage 6      (Papers + Projects + Production)
```

> **The most important rule**: *Build something with every concept you learn. Reading without coding is trivia, not knowledge.*

---

*Happy learning. The field moves fast — stay curious, stay building.* 🚀
