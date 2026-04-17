# gravitee-inference

**gravitee-inference** is a Java library designed to make it easy for engineering teams to integrate and deploy AI models within the Gravitee platform—without needing specialized help from AI/ML teams.

---

## Requirements

- Java 21
- Maven (`mvn`)

---

## Import libraries

In your `pom.xml` add the dependencies

```xml
<dependency>
  <groupId>io.gravitee.inference.math.native</groupId>
  <artifactId>gravitee-inference-math-native</artifactId>
  <version>${gravitee.inference.version}</version>
</dependency>

<dependency>
  <groupId>io.gravitee.inference.api</groupId>
  <artifactId>gravitee-inference-api</artifactId>
  <version>${gravitee.inference.version}</version>
</dependency>

<dependency>
  <groupId>io.gravitee.inference.onnx</groupId>
  <artifactId>gravitee-inference-onnx</artifactId>
  <version${gravitee.inference.version}</version>
</dependency>
```

## Supported AI Models

### BERT (via ONNX)

We support BERT architecture in ONNX format for various NLP tasks:

- Sequence Classification
- Token Classification
- Fill-mask
- Vector Embedding (e.g., Sentence Similarity)

---

### 🧠 Sequence Classification

Use this to determine sentiment or categorize full sentences.

```java
var resource = new OnnxBertResource(
    Paths.get("/path/to/your/model.onnx"),
    Paths.get("/path/to/your/tokenizer.json")
);

var configuration = Map.of(
    CLASSIFIER_MODE, ClassifierMode.SEQUENCE,
    CLASSIFIER_LABELS, List.of("Negative", "Positive")
);

var onnxConfig = new OnnxBertConfig(
    resource,
    NativeMath.INSTANCE,
    configuration
);

var model = new OnnxBertClassifierModel(onnxConfig);

// Single sentence
List<ClassifierResult> results = model.infer("I am so happy!").results();
results.forEach(result -> {
    System.out.println("Label: " + result.label());
    System.out.println("Score: " + result.score());
});

// Multiple sentences
model.infer(List.of("I am so happy!", "I am so sad!"));
```

> Try this with [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).

---

### 🧾 Token Classification

Use this to extract structured entities like names, locations, and organizations from text.

```java
var resource = new OnnxBertResource(
    Paths.get("/path/to/your/model.onnx"),
    Paths.get("/path/to/your/tokenizer.json")
);

var configuration = Map.of(
    Constants.CLASSIFIER_MODE, ClassifierMode.TOKEN,
    Constants.CLASSIFIER_LABELS, List.of(
        "O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"
    ),
    Constants.DISCARD_LABELS, List.of("O", "B-MISC", "I-MISC")
);

var onnxConfig = new OnnxBertConfig(resource, NativeMath.INSTANCE, configuration);
var model = new OnnxBertClassifierModel(onnxConfig);

List<ClassifierResult> results = model.infer("My name is Laura and I live in Houston, Texas").results();
results.forEach(result -> {
    System.out.println("Label: " + result.label());
    System.out.println("Score: " + result.score());
    System.out.println("Begin: " + result.begin());
    System.out.println("End: " + result.end());
});
```

```java
model.infer(List.of(
    "My name is Laura and I live in Houston, Texas",
    "My name is Clara and I live in Berkley, California"
));
```

> Try this with [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER/).

---

### 🎭 Fill Mask

Predict masked tokens in a sentence.

```java
var resource = new OnnxBertResource(
    Paths.get("/path/to/your/model.onnx"),
    Paths.get("/path/to/your/tokenizer.json")
);

var onnxConfig = new OnnxBertConfig(resource, NativeMath.INSTANCE, Map.of());
var model = new OnnxBertFillMaskInference(onnxConfig);

List<FillMaskResult> results = model.infer("The capital of France is [MASK].");

System.out.println(results.getFirst().label()); // Paris
```

```java
model.infer(List.of(
    "The capital of France is [MASK].",
    "The capital of [MASK] is London."
));
```

> Try this with [`google-bert/bert-base-uncased`](https://huggingface.co/google-bert/bert-base-uncased).

---

### 📐 Vector Embeddings

Convert text into dense vector representations for similarity search or indexing.

```java
var resource = new OnnxBertResource(
    Paths.get("/path/to/your/model.onnx"),
    Paths.get("/path/to/your/tokenizer.json")
);

var onnxConfig = new OnnxBertConfig(resource, NativeMath.INSTANCE, Map.of(
    POOLING_MODE, PoolingMode.MEAN,
    Constants.MAX_SEQUENCE_LENGTH, 512
));

var model = new OnnxBertEmbeddingModel(onnxConfig);
EmbeddingTokenCount embedding = model.infer("The big brown fox jumped over the lazy dog");

System.out.println(embedding.embedding().length); // 384
System.out.println(embedding.tokenCount()); // 11

// Similarity comparison
EmbeddingTokenCount embedding1 = model.infer("The big brown fox jumped over the lazy dog");
EmbeddingTokenCount embedding2 = model.infer("The brown fox jumped over the dog");

System.out.println(
    onnxConfig.gioMaths().cosineScore(embedding1.embedding(), embedding2.embedding())
);
```

> Try this with [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2).

---

### ⚡ SIMD Capabilities

To run with SIMD math acceleration:

1. Add the following to your JVM arguments:

```sh
--add-modules jdk.incubator.vector
```

2. Import the according dependencies:

```xml
<dependency>
    <groupId>io.gravitee.inference.math.simd</groupId>
    <artifactId>gravitee-inference-math-simd</artifactId>
    <version>${gravitee.inference.version}</version>
</dependency>
```

```java
import io.gravitee.inference.math.simd.factory.SIMDMathFactory;

GioMaths maths = SIMDMathFactory.gioMaths();
```

The factory will resolve at runtime which SIMD capability your CPU handles.

---

## Llama.cpp Inference (GGUF)

`gravitee-inference-llama-cpp` runs local LLM inference using [llama.cpp](https://github.com/ggml-org/llama.cpp) via Java's Foreign Function & Memory API. Models must be in **GGUF** format.

### ModelConfig Reference

All parameters are set through the `ModelConfig` record:

| Parameter | Type | Description |
|---|---|---|
| `modelPath` | `Path` | Absolute path to the `.gguf` model file. **Required.** |
| `nCtx` | `int` | Context window size (tokens). Larger values use more memory. Typical: `2048`–`8192`. |
| `nBatch` | `int` | Maximum tokens processed per batch during prompt evaluation. Higher = faster prompt ingestion but more memory. Typical: `512`–`2048`. |
| `nUBatch` | `int` | Micro-batch size (tokens per internal compute step). Usually same as `nBatch` or smaller. |
| `nSeqMax` | `int` | Maximum concurrent sequences (conversations) in the batch engine. |
| `nThreads` | `int` | CPU threads for inference. Use physical core count (not hyperthreads). |
| `nThreadsBatch` | `int` | CPU threads for batch prompt processing. Can be higher than `nThreads`. |
| `nGpuLayers` | `int` | Number of transformer layers offloaded to GPU. `0` = CPU-only, `99` = offload all. |
| `useMlock` | `boolean` | Lock model weights in RAM (prevents OS paging). Recommended for production. |
| `useMmap` | `boolean` | Memory-map model weights from disk. Faster loading, shared across processes. |
| `splitMode` | `SplitMode` | Multi-GPU split strategy: `NONE` (single GPU), `LAYER` (split by layers), `ROW` (tensor parallelism). |
| `mainGpu` | `int` | Primary GPU index for single-GPU or scratch buffer allocation. |
| `flashAttnType` | `FlashAttentionType` | Flash Attention: `AUTO` (let llama.cpp decide), `ENABLED`, or `DISABLED`. |
| `offloadKQV` | `boolean` | Offload KV-cache to GPU. Strongly recommended when using GPU layers. |
| `noPerf` | `boolean` | Disable performance counters. Set `false` to collect timing metrics. |
| `logLevel` | `LlamaLogLevel` | Native llama.cpp log verbosity. `null` to disable native logging. |
| `loraPath` | `Path` | Path to a LoRA adapter `.gguf` file, or `null`. |
| `mmprojPath` | `Path` | Path to a multimodal projector `.gguf` file (for vision/audio models), or `null`. |
| `rpcServers` | `List<String>` | RPC server endpoints (`"host:port"`) for distributed inference. Empty or `null` for local-only. |
| `memoryCheckPolicy` | `MemoryCheckPolicy` | Pre-flight memory check: `FAIL` (abort if model won't fit), `WARN` (log and continue), `DISABLED` (skip). |

### Memory Tuning

The engine runs a pre-flight memory check before loading weights. The log output looks like:

```
INFO  EngineAdapter - Running memory pre-flight check for Qwen3-0.6B-Q8_0.gguf (policy=WARN)
INFO  EngineAdapter - Memory pre-flight: VRAM estimate (exact): required=0.65 GiB, available=25.47 GiB — fits. Model fits with 97% headroom.
```

If the model doesn't fit, the estimator suggests a reduced `nGpuLayers` value:

```
INFO  EngineAdapter - Memory pre-flight: VRAM estimate (exact): required=14.20 GiB, available=8.00 GiB — does NOT fit.
                      Try nGpuLayers=12 (requested 99) to fit within available VRAM.
```

**What the estimator accounts for:**
- Model weights (proportional to `nGpuLayers` / total layers)
- KV-cache (`nCtx` x layers x heads x head_dim x 4 bytes per token)
- Multimodal projector weights (if `mmprojPath` is set)
- LoRA adapter weights (if `loraPath` is set)
- 10% GPU safety margin for driver/OS overhead

**What it does NOT account for:**
- Compute graph scratch buffers (allocated at first inference)
- Vision encoder activation memory (temporary, varies with image resolution)
- System memory used by the GGUF file mapping

#### Choosing `nGpuLayers`

| Scenario | Recommendation |
|---|---|
| Model fits entirely in VRAM | Set `nGpuLayers` to `99` (offload all layers) |
| Model partially fits | Use the suggested layer count from the memory check log |
| No GPU / CPU-only | Set `nGpuLayers` to `0` |
| Distributed (RPC) | Set `nGpuLayers` to `99` — layers are distributed across RPC servers |

#### Choosing `nCtx`

The context window directly impacts KV-cache memory. Each token costs:

```
bytes_per_token = nLayers x nHeadKv x headDim x 4
```

For a typical 1B model (16 layers, 8 KV-heads, 64 head_dim): **16 KB per token**.
For a 7B model (32 layers, 8 KV-heads, 128 head_dim): **128 KB per token**.

| `nCtx` | KV-cache (1B model) | KV-cache (7B model) |
|---|---|---|
| 2048 | 32 MiB | 256 MiB |
| 4096 | 64 MiB | 512 MiB |
| 8192 | 128 MiB | 1 GiB |
| 32768 | 512 MiB | 4 GiB |

Start with `nCtx=4096` and increase only if your use case requires long conversations.

#### Choosing `nBatch` and `nUBatch`

- `nBatch` controls how many tokens are processed per prompt evaluation step. Larger batches = faster prompt ingestion but more memory.
- `nUBatch` is the micro-batch size used internally. Set equal to `nBatch` unless memory-constrained.

| Scenario | `nBatch` | `nUBatch` |
|---|---|---|
| Low memory | `256` | `256` |
| Balanced | `512` | `512` |
| Maximum throughput | `2048` | `2048` |

#### Threading

- `nThreads`: set to the number of **physical** CPU cores (not hyperthreads). Over-subscribing hurts performance.
- `nThreadsBatch`: can be higher than `nThreads` since batch processing is less latency-sensitive.

#### Flash Attention

Flash Attention reduces memory usage for the KV-cache and improves throughput. Use `AUTO` (default) to let llama.cpp decide based on the model architecture and hardware. Force `ENABLED` if you want to ensure it's on.

#### `useMlock` vs `useMmap`

| Setting | Behavior | When to use |
|---|---|---|
| `useMmap=true` | Memory-maps the model file; pages loaded on demand | Default. Good for development and shared environments. |
| `useMlock=true` | Locks all model pages in physical RAM after loading | Production. Prevents OS from paging out weights under memory pressure. |
| Both `true` | Model is memory-mapped then locked | Best for production with large models — fast loading + guaranteed resident memory. |

#### RPC (Distributed Inference)

When `rpcServers` is configured, model layers are distributed across remote GPU servers:

1. Set `nGpuLayers=99` — layers are split across all servers proportionally to their free VRAM.
2. The memory estimator queries each server's free memory via the RPC protocol (no backend registration needed).
3. The bottleneck is the server with the **least free VRAM** — the estimator reports this as available memory.

#### Memory Check Policy

| Policy | Behavior |
|---|---|
| `WARN` | Log estimate at INFO, warn if model won't fit, continue loading. **Recommended for production.** |
| `FAIL` | Log estimate at INFO, throw `InsufficientVramException` if model won't fit. Use in CI or strict environments. |
| `DISABLED` | Skip the check entirely. Use when you know the model fits or the estimator is not needed. |

---

## vLLM Inference (HuggingFace models)

`gravitee-inference-vllm` runs LLM inference using [vLLM](https://github.com/vllm-project/vllm) through an embedded CPython interpreter (via [vLLM4j](https://github.com/gravitee-io/vLLM4j)). Models are loaded directly from HuggingFace Hub — no GGUF conversion required.

### Requirements

- Linux with CUDA GPU (Metal not supported)
- A Python virtual environment with vLLM installed:
  ```sh
  python -m venv .venv
  source .venv/bin/activate
  pip install vllm
  ```
- Pass the venv path via `-Dvllm4j.venv=/path/to/.venv` or `VLLM_VENV_PATH` env var

### VllmConfig Reference

| Parameter | Type | Description |
|---|---|---|
| `model` | `String` | HuggingFace model name (e.g. `Qwen/Qwen3-0.6B`). **Required.** |
| `dtype` | `String` | Weight data type: `auto`, `float16`, `bfloat16`, `float32`. |
| `quantization` | `String` | Quantization method: `gptq`, `awq`, `fp8`, or `null` (no quant). |
| `maxModelLen` | `int` | Maximum sequence length (context window). |
| `maxNumSeqs` | `int` | Maximum concurrent sequences. |
| `gpuMemoryUtilization` | `float` | Fraction of GPU memory to use (0.0–1.0). |
| `enforceEager` | `boolean` | Disable CUDA graphs for reduced memory. |
| `trustRemoteCode` | `boolean` | Allow executing model-provided Python code. |
| `enableChunkedPrefill` | `boolean` | Enable chunked prefill for long prompts. |
| `enablePrefixCaching` | `boolean` | Cache shared prefixes across requests. |
| `enableSleepMode` | `boolean` | Put the engine to sleep when idle. |
| `memoryCheckPolicy` | `MemoryCheckPolicy` | Pre-flight memory check: `FAIL`, `WARN`, or `DISABLED`. |

### Quick Start

```java
var config = VllmConfig.builder()
    .model("Qwen/Qwen3-0.6B")
    .dtype("auto")
    .maxModelLen(4096)
    .maxNumSeqs(1)
    .gpuMemoryUtilization(0.35f)
    .enforceEager(true)
    .trustRemoteCode(true)
    .build();

var engine = new BatchEngine(config);
engine.start(token -> {
    System.out.print(token.token());
    if (token.isFinal()) System.out.println();
});

var request = VllmRequest.builder()
    .prompt("Hello, how are you?")
    .maxTokens(256)
    .temperature(0.7f)
    .build();

engine.addSequence(0, request);
```

### Chat Template

The model's Jinja2 chat template is accessible via:

```java
String template = engine.chatTemplateString(); // from HuggingFace tokenizer_config.json
String bos = engine.bosToken();
String eos = engine.eosToken();
```