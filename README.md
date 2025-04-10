Sure! Here's your content rewritten in proper Markdown format:

---

# gravitee-inference

**gravitee-inference** is a Java library designed to make it easy for engineering teams to integrate and deploy AI models within the Gravitee platform—without needing specialized help from AI/ML teams.

---

## Requirements

- Java 21
- Maven (`mvn`)

---

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
List<ClassifierResult> results = model.infer("I am so happy!");
results.forEach(result -> {
    System.out.println("Label: " + result.label());
    System.out.println("Score: " + result.score());
});

// Multiple sentences
List<List<ClassifierResult>> batchResults = model.infer(List.of("I am so happy!", "I am so sad!"));
batchResults.stream().flatMap(List::stream).forEach(result -> {
    System.out.println("Label: " + result.label());
    System.out.println("Score: " + result.score());
});
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

List<ClassifierResult> results = model.infer("My name is Laura and I live in Houston, Texas");
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

System.out.println(embedding.embeddings().length); // 384
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

To run Gravitee Gateway with SIMD math acceleration:

Add the following to your JVM arguments:

```sh
--add-modules jdk.incubator.vector
```