//! xazz-exec/src/dl.rs — Burn deep-learning execution engine (v0.4)
//!
//! Translates `.xzz` `model {}` declarations and `run v |> train(...)` syntax into
//! actual Burn neural-network training and executes it.
//!
//! Backend: burn-ndarray (pure Rust CPU). Structured around the `Backend` generic,
//! so switching to torch / wgpu only requires swapping this module's `AD`/`Plain` type aliases.
//!
//!   - AD    = NdArrayAutodiff<f32>  (for training: autodiff graph enabled)
//!   - Plain = NdArray<f32>          (for inference)

use std::collections::HashMap;

use burn::{
    backend::Autodiff,
    module::{AutodiffModule, Module},
    nn::{
        DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
        conv::{Conv1d, Conv1dConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, PrettyJsonFileRecorder},
    tensor::{
        Device, Tensor, TensorData,
        activation::{relu, sigmoid, softmax, tanh},
        backend::{AutodiffBackend, Backend},
    },
};
use burn_ndarray::NdArray;
use polars::prelude::{Column, DataFrame};
use xazz_compiler::ast::{LayerKind, TrainConfig};
use xazz_core::i18n::{is_korean, tr};

use crate::tensor_bridge::{extract_data, series_to_f32};

/// Autodiff backend for training (CPU): NdArray + Autodiff wrapper.
pub type AD = Autodiff<NdArray<f32>>;
/// Pure backend for inference (CPU).
pub type Plain = NdArray<f32>;

/// One training batch on an autodiff backend: `(features, targets)`.
type TrainBatch<B> = (Tensor<Autodiff<B>, 2>, Tensor<Autodiff<B>, 2>);

/// Upper bound of the validation split ratio — at most this fraction of the data can be held out for validation.
const MAX_VALIDATION_SPLIT: f64 = 0.9;

/// DSL model block activation/normalization layer kinds.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Activation {
    None,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout(f64),
}

/// Applies the per-layer activation function.
/// Dropout is applied only during training (training=true); inference passes it through as the identity function.
fn apply_activation<B: Backend, const D: usize>(
    act: &Activation,
    x: Tensor<B, D>,
    training: bool,
) -> Tensor<B, D> {
    match act {
        Activation::None => x,
        Activation::ReLU => relu(x),
        Activation::Sigmoid => sigmoid(x),
        Activation::Tanh => tanh(x),
        Activation::Softmax => softmax(x, 1),
        Activation::Dropout(prob) => {
            if training {
                DropoutConfig::new(*prob).init().forward(x)
            } else {
                x
            }
        }
    }
}

/// A single forward operation in the compiled model graph, in declaration order.
/// The payload indexes into [`Mlp::linears`] / [`Mlp::convs`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum LayerOp {
    /// Dense (Linear) layer.
    Dense(usize),
    /// Conv1d layer (1D convolution over the feature axis).
    Conv1d(usize),
    /// Embedding layer (categorical index → vector), always the first op.
    Embedding(usize),
}

/// Burn module expressing the DSL `model { Dense -> ReLU -> ... }` as a dynamic
/// graph of Dense / Conv1d layers — a multi-layer perceptron (MLP) and/or a 1D
/// convolutional network over tabular features.
// (The Burn Module derive provides Clone)
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    /// Sequence of Dense(units) layers.
    linears: Vec<Linear<B>>,
    /// Sequence of Conv1d(out_channels, kernel_size) layers.
    convs: Vec<Conv1d<B>>,
    /// Sequence of Embedding(vocab, embed_dim) layers.
    embeddings: Vec<Embedding<B>>,
    /// Per-embedding, per-column vocabulary size (for index clamping); parallel
    /// to `embeddings`.
    #[module(skip)]
    embed_vocab: Vec<Vec<usize>>,
    /// Per-embedding, per-column row offset into the combined table; parallel to
    /// `embeddings` and `embed_vocab`.
    #[module(skip)]
    embed_offsets: Vec<Vec<usize>>,
    /// Forward order: each entry pairs an op with the activation applied after it.
    #[module(skip)]
    ops: Vec<(LayerOp, Activation)>,
    /// Output feature dimension (result of the declared graph) — reported in the train report.
    #[module(skip)]
    out_dim: usize,
    /// Whether the graph consumes raw category indices (first layer is Embedding).
    #[module(skip)]
    raw_input: bool,
    /// Whether in training mode — determines Dropout application (false during inference).
    #[module(skip)]
    training: bool,
}

impl<B: Backend> Mlp<B> {
    /// Forward pass: [batch, input_dim] → [batch, output_dim].
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;
        for (op, act) in &self.ops {
            x = match op {
                LayerOp::Dense(i) => {
                    apply_activation(act, self.linears[*i].forward(x), self.training)
                }
                LayerOp::Conv1d(i) => {
                    // [batch, len] → [batch, 1, len] → conv (Same padding keeps len) → [batch, channels, len]
                    let [batch, len] = x.dims();
                    let y = self.convs[*i].forward(x.reshape([batch, 1, len]));
                    let [b, c, l] = y.dims();
                    apply_activation(act, y.reshape([b, c * l]), self.training)
                }
                LayerOp::Embedding(i) => {
                    // [batch, len] category indices → [batch, len, embed_dim] → [batch, len * embed_dim].
                    // Each input column j has its own vocab and row offset in the combined table.
                    let vocabs = &self.embed_vocab[*i];
                    let offsets = &self.embed_offsets[*i];
                    let [_, len] = x.dims();
                    let mut cols: Vec<Tensor<B, 2>> = Vec::with_capacity(len);
                    for (j, (&v, &offset)) in vocabs.iter().zip(offsets).enumerate() {
                        let idx = x
                            .clone()
                            .narrow(1, j, 1)
                            .clamp(0.0, v.saturating_sub(1) as f32)
                            + offset as f32;
                        cols.push(idx);
                    }
                    let idx = Tensor::cat(cols, 1).int();
                    let y = self.embeddings[*i].forward(idx);
                    let [b, l, d] = y.dims();
                    apply_activation(act, y.reshape([b, l * d]), self.training)
                }
            };
        }
        x
    }
}

/// Builds the MLP/CNN from the DSL layer list and input dimension.
fn build_mlp<B: Backend>(
    layers: &[LayerKind],
    input_dim: usize,
    device: &Device<B>,
) -> Result<Mlp<B>, String> {
    let mut linears: Vec<Linear<B>> = Vec::new();
    let mut convs: Vec<Conv1d<B>> = Vec::new();
    let mut embeddings: Vec<Embedding<B>> = Vec::new();
    let mut embed_vocab: Vec<Vec<usize>> = Vec::new();
    let mut embed_offsets: Vec<Vec<usize>> = Vec::new();
    let mut ops: Vec<(LayerOp, Activation)> = Vec::new();
    let mut cur = input_dim;

    for layer in layers {
        match layer {
            LayerKind::Dense(n) if *n > 0 => {
                linears.push(LinearConfig::new(cur, *n).init(device));
                ops.push((LayerOp::Dense(linears.len() - 1), Activation::None));
                cur = *n;
            }
            LayerKind::Dense(_) => {
                return Err(tr(
                    "Dense layer unit count must be >= 1.",
                    "Dense 레이어의 유닛 수는 1 이상이어야 합니다.",
                )
                .into());
            }
            LayerKind::Conv1d {
                out_channels,
                kernel_size,
            } if *out_channels > 0 && *kernel_size > 0 => {
                let config = Conv1dConfig::new(1, *out_channels, *kernel_size)
                    .with_padding(PaddingConfig1d::Same);
                convs.push(config.init(device));
                ops.push((LayerOp::Conv1d(convs.len() - 1), Activation::None));
                // Same padding + stride 1 preserves the length: [batch, 1, cur] → [batch, out_channels, cur].
                cur *= *out_channels;
            }
            LayerKind::Conv1d { .. } => {
                return Err(tr(
                    "Conv1d out_channels and kernel_size must be >= 1.",
                    "Conv1d 의 out_channels 와 kernel_size 는 1 이상이어야 합니다.",
                )
                .into());
            }
            LayerKind::Embedding { vocab, embed_dim } if vocab.is_valid() && *embed_dim > 0 => {
                // One combined table per embedding layer; input column j owns the
                // disjoint row range [offset_j, offset_j + vocab_j).
                let sizes = vocab.expand(cur)?;
                let total: usize = sizes.iter().sum();
                if total == 0 {
                    return Err(tr(
                        "Embedding has no vocabulary entries to embed.",
                        "Embedding 에 임베딩할 vocab 항목이 없습니다.",
                    )
                    .into());
                }
                embeddings.push(EmbeddingConfig::new(total, *embed_dim).init(device));
                let mut offsets = Vec::with_capacity(sizes.len());
                let mut offset = 0usize;
                for size in &sizes {
                    offsets.push(offset);
                    offset += size;
                }
                ops.push((LayerOp::Embedding(embeddings.len() - 1), Activation::None));
                embed_vocab.push(sizes);
                embed_offsets.push(offsets);
                // Each of the `cur` input positions is embedded into `embed_dim` features.
                cur *= *embed_dim;
            }
            LayerKind::Embedding { .. } => {
                return Err(tr(
                    "Embedding vocab and embed_dim must be >= 1.",
                    "Embedding 의 vocab 과 embed_dim 은 1 이상이어야 합니다.",
                )
                .into());
            }
            LayerKind::ReLU => set_activation(&mut ops, Activation::ReLU),
            LayerKind::Sigmoid => set_activation(&mut ops, Activation::Sigmoid),
            LayerKind::Tanh => set_activation(&mut ops, Activation::Tanh),
            LayerKind::Softmax => set_activation(&mut ops, Activation::Softmax),
            LayerKind::Dropout(r) => set_activation(&mut ops, Activation::Dropout(*r)),
            // BatchNorm (1D MLP) is omitted because its layout differs from Burn's 2D BatchNorm.
            LayerKind::BatchNorm => { /* pass-through */ }
        }
    }

    if ops.is_empty() {
        return Err(tr(
            "The model has no Dense or Conv1d layers.",
            "모델에 Dense 또는 Conv1d 레이어가 하나도 없습니다.",
        )
        .into());
    }
    Ok(Mlp {
        linears,
        convs,
        embeddings,
        embed_vocab,
        embed_offsets,
        ops,
        out_dim: cur,
        raw_input: matches!(layers.first(), Some(LayerKind::Embedding { .. })),
        training: true,
    })
}

/// Records the activation after the last Dense/Conv1d op (consecutive activations keep only the last one).
fn set_activation(ops: &mut [(LayerOp, Activation)], act: Activation) {
    if let Some(last) = ops.last_mut() {
        last.1 = act;
    }
}

/// Per-column vocabulary sizes of the leading Embedding layer, if the model
/// consumes raw category indices. The checker enforces Embedding-first, so there
/// is at most one such layer.
fn leading_embedding_vocabs(layers: &[LayerKind], feature_count: usize) -> Option<Vec<usize>> {
    match layers.first() {
        Some(LayerKind::Embedding { vocab, .. }) if vocab.is_valid() => {
            vocab.expand(feature_count).ok()
        }
        _ => None,
    }
}

/// Counts raw embedding indices outside `[0, vocab_size - 1]`. The forward pass
/// clamps them silently, so this feeds the runtime diagnostic below (issue D3).
/// Non-finite values are excluded: the forward pass maps them to index 0.
fn count_out_of_range_indices(values: &[f32], vocab_size: usize) -> usize {
    let max = vocab_size.saturating_sub(1) as f32;
    values
        .iter()
        .filter(|v| v.is_finite() && (**v < 0.0 || **v > max))
        .count()
}

/// Counts raw embedding inputs that are finite but not whole numbers. The
/// forward pass casts the raw value to an integer (truncation), so a continuous
/// feature fed to Embedding silently loses its fractional part — this feeds the
/// runtime diagnostic below (issue D3). Non-finite values are excluded because
/// the forward pass maps them to index 0.
fn count_non_integer_indices(values: &[f32]) -> usize {
    values
        .iter()
        .filter(|v| v.is_finite() && v.fract() != 0.0)
        .count()
}

/// Counts out-of-range indices across all columns, each against its own vocab.
fn count_out_of_range_per_column(values: &[f32], feature_count: usize, vocabs: &[usize]) -> usize {
    if feature_count == 0 {
        return 0;
    }
    let mut count = 0usize;
    for (idx, v) in values.iter().enumerate() {
        if let Some(&vocab) = vocabs.get(idx % feature_count) {
            count += count_out_of_range_indices(std::slice::from_ref(v), vocab);
        }
    }
    count
}

/// Emits the out-of-range embedding diagnostic to stderr. Non-fatal — the value
/// is clamped, matching the documented forward-pass behaviour.
fn warn_embedding_out_of_range(count: usize, vocabs: &[usize]) {
    let uniform = vocabs.windows(2).all(|w| w[0] == w[1]);
    let msg = if uniform {
        let vocab_size = vocabs.first().copied().unwrap_or(0);
        let max = vocab_size.saturating_sub(1);
        if is_korean() {
            format!(
                "Embedding 입력 범주 인덱스 {count}개가 범위를 벗어났습니다 (vocab_size={vocab_size}). forward 에서 [0, {max}] 로 clamp 됩니다. 범주형 컬럼 값/스키마를 확인하세요."
            )
        } else {
            format!(
                "{count} embedding input index/indices are out of range (vocab_size={vocab_size}); they are clamped to [0, {max}] in the forward pass. Check the categorical column values/schema."
            )
        }
    } else {
        let list = vocabs
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        if is_korean() {
            format!(
                "Embedding 입력 범주 인덱스 {count}개가 범위를 벗어났습니다 (컬럼별 vocab=[{list}]). forward 에서 컬럼별 [0, vocab-1] 로 clamp 됩니다. 범주형 컬럼 값/스키마를 확인하세요."
            )
        } else {
            format!(
                "{count} embedding input index/indices are out of range (per-column vocab=[{list}]); they are clamped to each column's [0, vocab-1] in the forward pass. Check the categorical column values/schema."
            )
        }
    };
    eprintln!("[xazz] {msg}");
}

/// Emits the non-integer embedding diagnostic to stderr. Non-fatal — the value
/// is truncated, matching the documented forward-pass behaviour. A continuous
/// feature must not be fed to Embedding: z-score is skipped for raw-index models,
/// so the raw value becomes an index and its fractional part is lost.
fn warn_embedding_non_integer(count: usize) {
    let msg = if is_korean() {
        format!(
            "Embedding 입력 {count}개가 정수가 아닌 연속형 값입니다. raw 인덱스 모델은 z-score 를 건너뛰므로 소수부가 버려져(truncate) 인덱스로 사용됩니다. 범주형(정수 인코딩) 컬럼인지 확인하세요."
        )
    } else {
        format!(
            "{count} embedding input value(s) are non-integer continuous values; raw-index models skip z-score, so the fractional part is truncated to form an index. Check that the columns are categorical (integer-coded)."
        )
    };
    eprintln!("[xazz] {msg}");
}

/// Runs the out-of-range embedding diagnostic when the model consumes raw indices.
fn check_embedding_indices(layers: &[LayerKind], values: &[f32], feature_count: usize) {
    if let Some(vocabs) = leading_embedding_vocabs(layers, feature_count) {
        let count = count_out_of_range_per_column(values, feature_count, &vocabs);
        if count > 0 {
            warn_embedding_out_of_range(count, &vocabs);
        }
        let non_integer = count_non_integer_indices(values);
        if non_integer > 0 {
            warn_embedding_non_integer(non_integer);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training result report
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainReport {
    pub model_name: String,
    pub target: String,
    pub feature_names: Vec<String>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub num_params: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub predictions: Vec<f64>,
    pub targets: Vec<f64>,
    pub checkpoint_path: String,
    /// Whether training stopped early (validation loss plateaued) — issue D3.
    #[serde(default)]
    pub stopped_early: bool,
    /// Best validation epoch (1-based); 0 when no validation split was used.
    #[serde(default)]
    pub best_epoch: usize,
}

/// Trained model — also holds the standardization statistics needed for predict().
#[derive(Debug, Clone)]
pub struct TrainedModel {
    /// Pure model for inference (no autodiff graph).
    pub model: Mlp<Plain>,
    /// Training result report (for markers/logs).
    pub report: TrainReport,
    /// Declared layer graph, kept so a non-CPU provider can rebuild the module
    /// structure and load the portable checkpoint onto its own device.
    pub layers: Vec<LayerKind>,
    /// Feature column order (1:1 correspondence with standardization statistics).
    pub feature_names: Vec<String>,
    /// Per-feature mean (z-score).
    pub fmean: Vec<f64>,
    /// Per-feature standard deviation (z-score).
    pub fstd: Vec<f64>,
    /// Training target column.
    pub target: String,
}

/// One evaluated point of a hyperparameter sweep (D3).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepCombo {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub final_train_loss: f64,
    pub final_val_loss: Option<f64>,
    pub stopped_early: bool,
    pub best_epoch: usize,
    /// Whether this combination was selected as the sweep winner.
    pub selected: bool,
}

/// Result of a grid-search hyperparameter sweep (D3).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SweepReport {
    pub model_name: String,
    pub target: String,
    pub combos: Vec<SweepCombo>,
    /// Index into `combos` of the selected (best) combination.
    pub best_index: usize,
}

impl SweepReport {
    /// Selection metric: validation loss when available, else training loss.
    /// Non-finite losses rank last.
    pub fn score(combo: &SweepCombo) -> f64 {
        match combo.final_val_loss {
            Some(v) if v.is_finite() => v,
            _ if combo.final_train_loss.is_finite() => combo.final_train_loss,
            _ => f64::INFINITY,
        }
    }
}

/// Persists an inference model to `path` (Burn appends the `.json` extension).
pub fn save_checkpoint(model: &Mlp<Plain>, path: &str) -> Result<(), String> {
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(path, &recorder).map_err(|e| {
        format!(
            "{}: {e}",
            tr("checkpoint save failed", "체크포인트 저장 실패")
        )
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Backend-agnostic training output: a plain (non-autodiff) module on backend `B`
/// plus everything needed to build the portable [`TrainedModel`] artifact.
struct RawTrained<B: Backend> {
    model: Mlp<B>,
    report: TrainReport,
    feature_names: Vec<String>,
    fmean: Vec<f64>,
    fstd: Vec<f64>,
    target: String,
}

/// Runs `dataset |> train(<model>, target: "...", ...)` on the default CPU backend.
pub fn train(
    df: &DataFrame,
    model_name: &str,
    layers: &[LayerKind],
    config: &TrainConfig,
) -> Result<TrainedModel, String> {
    let raw = train_impl::<NdArray<f32>>(df, model_name, layers, config)?;
    Ok(TrainedModel {
        model: raw.model,
        report: raw.report,
        layers: layers.to_vec(),
        feature_names: raw.feature_names,
        fmean: raw.fmean,
        fstd: raw.fstd,
        target: raw.target,
    })
}

/// Trains on an arbitrary Burn backend `B`, then materialises the portable CPU
/// [`TrainedModel`] artifact by round-tripping the checkpoint through Burn's
/// backend-neutral record format. GPU providers (e.g. `burn-wgpu`) call this so
/// the artifact they hand back is structurally identical to the CPU reference,
/// and `predict`/`predict_on` can consume it on any device.
pub fn train_on<B>(
    df: &DataFrame,
    model_name: &str,
    layers: &[LayerKind],
    config: &TrainConfig,
) -> Result<TrainedModel, String>
where
    B: Backend,
    Autodiff<B>: AutodiffBackend,
    Mlp<Autodiff<B>>: AutodiffModule<Autodiff<B>, InnerModule = Mlp<B>>,
{
    let raw = train_impl::<B>(df, model_name, layers, config)?;
    let device: Device<NdArray<f32>> = Default::default();
    let template = build_mlp::<NdArray<f32>>(layers, raw.report.input_dim, &device)?;
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    let mut model = template
        .load_file(raw.report.checkpoint_path.as_str(), &recorder, &device)
        .map_err(|e| {
            format!(
                "{}: {e}",
                tr("checkpoint load failed", "체크포인트 로드 실패")
            )
        })?;
    model.training = false;
    Ok(TrainedModel {
        model,
        report: raw.report,
        layers: layers.to_vec(),
        feature_names: raw.feature_names,
        fmean: raw.fmean,
        fstd: raw.fstd,
        target: raw.target,
    })
}

/// Backend-agnostic training core: trains `Mlp<B>` and returns its report and stats.
fn train_impl<B>(
    df: &DataFrame,
    model_name: &str,
    layers: &[LayerKind],
    config: &TrainConfig,
) -> Result<RawTrained<B>, String>
where
    B: Backend,
    Autodiff<B>: AutodiffBackend,
    Mlp<Autodiff<B>>: AutodiffModule<Autodiff<B>, InnerModule = Mlp<B>>,
{
    let (feature_names, features, targets) = extract_data(df, &config.target)?;
    let n = features.len();
    let input_dim = feature_names.len();
    // Embedding-first models consume raw category indices — z-score normalization
    // would destroy category identity, so keep raw values (NaN → 0).
    let raw_input = matches!(layers.first(), Some(LayerKind::Embedding { .. }));
    if input_dim == 0 {
        return Err(tr(
            "No numeric feature columns available for training.",
            "학습 가능한 숫자형 특성(컬럼)이 없습니다.",
        )
        .into());
    }
    if n == 0 {
        return Err(tr("Training data is empty.", "학습 데이터가 비어 있습니다.").into());
    }

    // ── Feature standardization statistics (NaN → mean imputation) ────────────────
    let mut fmean = vec![0f64; input_dim];
    let mut fstd = vec![1f64; input_dim];
    for (j, mean) in fmean.iter_mut().enumerate().take(input_dim) {
        let (mut s, mut c) = (0f64, 0usize);
        for i in 0..n {
            if let Some(v) = features.get(i).and_then(|row| row.get(j)) {
                let v = *v as f64;
                if v.is_finite() {
                    s += v;
                    c += 1;
                }
            }
        }
        *mean = if c > 0 { s / c as f64 } else { 0.0 };
    }
    for j in 0..input_dim {
        let (mut s, mut c) = (0f64, 0usize);
        for i in 0..n {
            if let Some(v) = features.get(i).and_then(|row| row.get(j)) {
                let d = *v as f64 - fmean[j];
                if d.is_finite() {
                    s += d * d;
                    c += 1;
                }
            }
        }
        fstd[j] = if c > 1 {
            (s / (c - 1) as f64).max(1e-8).sqrt()
        } else {
            1.0
        };
    }

    let mut xs = Vec::with_capacity(n * input_dim);
    for row in features.iter().take(n) {
        for j in 0..input_dim {
            let v = row[j] as f64;
            if raw_input {
                xs.push(if v.is_finite() { v as f32 } else { 0.0 });
            } else {
                let v = if v.is_finite() { v } else { fmean[j] };
                xs.push(((v - fmean[j]) / fstd[j]) as f32);
            }
        }
    }
    if raw_input {
        check_embedding_indices(layers, &xs, input_dim);
    }

    let tmean: f64 = {
        let (mut s, mut c) = (0f64, 0usize);
        for &t in &targets {
            if t.is_finite() {
                s += t as f64;
                c += 1;
            }
        }
        if c > 0 { s / c as f64 } else { 0.0 }
    };
    let ys: Vec<f32> = targets
        .iter()
        .map(|&t| if t.is_finite() { t } else { tmean as f32 })
        .collect();

    // ── train / validation split ────────────────────────────────────────────
    let val_split = config
        .validation_split
        .unwrap_or(0.0)
        .clamp(0.0, MAX_VALIDATION_SPLIT);
    let val_n = (n as f64 * val_split) as usize;
    let train_n = n - val_n;
    let val_idx: Vec<usize> = (train_n..n).collect();

    let device: Device<Autodiff<B>> = Default::default();
    let mut model = build_mlp::<Autodiff<B>>(layers, input_dim, &device)?;
    // A regression target is a single scalar; a model that ends in Conv1d/Embedding
    // (or a multi-unit Dense without a final Dense(1)) produces several outputs.
    // Fail closed instead of silently broadcasting the target (which then breaks
    // predict() with a column-length error).
    if model.out_dim != 1 {
        return Err(if is_korean() {
            format!(
                "모델 '{model_name}' 의 출력 차원이 {} 입니다. 회귀 타겟은 스칼라 하나여야 합니다. 마지막에 Dense(1) 을 추가하세요.",
                model.out_dim
            )
        } else {
            format!(
                "Model '{model_name}' outputs {} values; a regression target must be a single scalar. Add a final Dense(1).",
                model.out_dim
            )
        });
    }

    let batch_size = config.batch_size.unwrap_or(train_n.max(1));
    let lr = config.learning_rate;
    let mut optim = AdamConfig::new().init::<Autodiff<B>, _>();

    let make_batch = |idx: &[usize]| -> Option<TrainBatch<B>> {
        if idx.is_empty() {
            return None;
        }
        let b = idx.len();
        let mut xv = Vec::with_capacity(b * input_dim);
        let mut yv = Vec::with_capacity(b);
        for &i in idx {
            for j in 0..input_dim {
                xv.push(xs[i * input_dim + j]);
            }
            yv.push(ys[i]);
        }
        let x = Tensor::<Autodiff<B>, 2>::from_data(TensorData::new(xv, [b, input_dim]), &device);
        let y = Tensor::<Autodiff<B>, 2>::from_data(TensorData::new(yv, [b, 1]), &device);
        Some((x, y))
    };

    let mut final_train_loss = f64::NAN;
    let mut final_val_loss: Option<f64> = None;

    // Early stopping state (issue D3) — needs a validation split.
    let patience = config.early_stopping_patience.filter(|p| *p > 0);
    let mut best_val_loss = f64::INFINITY;
    let mut best_epoch: usize = 0;
    let mut epochs_no_improve = 0usize;
    let mut stopped_early = false;

    for epoch in 0..config.epochs {
        // Deterministic shuffle (epoch seed)
        let mut order: Vec<usize> = (0..train_n).collect();
        let mut seed = epoch as u64 + 1;
        for i in (1..order.len()).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (seed >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        let mut epoch_loss = 0f64;
        let mut steps = 0usize;
        for chunk in order.chunks(batch_size) {
            let (x, y) = match make_batch(chunk) {
                Some(v) => v,
                None => continue,
            };
            let out = model.forward(x);
            let loss = ((out - y).powf_scalar(2.0)).mean();
            let loss_val = loss
                .clone()
                .into_data()
                .to_vec::<f32>()
                .map(|v| v[0] as f64)
                .unwrap_or(f64::NAN);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(lr, model, grads);

            epoch_loss += loss_val;
            steps += 1;
        }
        final_train_loss = if steps > 0 {
            epoch_loss / steps as f64
        } else {
            f64::NAN
        };

        if !val_idx.is_empty()
            && let Some((xv, yv)) = make_batch(&val_idx)
        {
            let vout = model.forward(xv);
            let vloss = ((vout - yv).powf_scalar(2.0)).mean();
            final_val_loss = vloss.into_data().to_vec::<f32>().map(|v| v[0] as f64).ok();
        }

        // Early stopping: track the best validation loss and count epochs
        // without improvement (issue D3).
        if let Some(v) = final_val_loss {
            if v < best_val_loss {
                best_val_loss = v;
                best_epoch = epoch + 1;
                epochs_no_improve = 0;
            } else {
                epochs_no_improve += 1;
            }
        }

        let val_line = final_val_loss
            .map(|v| format!("  val_loss = {v:.6}"))
            .unwrap_or_default();
        println!(
            "  [Epoch {:>3}/{}]  train_loss = {final_train_loss:.6}{val_line}",
            epoch + 1,
            config.epochs
        );

        if let Some(p) = patience
            && !val_idx.is_empty()
            && epochs_no_improve >= p
        {
            println!(
                "  [Early stop] no val_loss improvement for {p} epoch(s); \
                     best epoch {best_epoch} (val_loss = {best_val_loss:.6})"
            );
            stopped_early = true;
            break;
        }
    }

    // ── Sample predictions (in-sample) ──────────────────────────────────────
    let n_pred = n.min(10);
    let device_plain: Device<B> = Default::default();
    let mut xv = Vec::with_capacity(n_pred * input_dim);
    for i in 0..n_pred {
        for j in 0..input_dim {
            xv.push(xs[i * input_dim + j]);
        }
    }
    let xp = Tensor::<B, 2>::from_data(TensorData::new(xv, [n_pred, input_dim]), &device_plain);
    let mut valid_model: Mlp<B> = model.valid();
    valid_model.training = false;
    let pred_t = valid_model.forward(xp);
    let preds = pred_t.into_data().to_vec::<f32>().unwrap_or_default();
    let predictions: Vec<f64> = preds.iter().map(|&v| v as f64).collect();
    let targets_out: Vec<f64> = (0..n_pred).map(|i| ys[i] as f64).collect();

    let num_params = model.num_params();
    let output_dim = model.out_dim;

    // ── Checkpoint save ────────────────────────────────────────────────────
    let ckpt_dir = "checkpoints";
    std::fs::create_dir_all(ckpt_dir).map_err(|e| {
        format!(
            "{}: {e}",
            tr(
                "failed to create checkpoints/ directory",
                "checkpoints/ 디렉토리 생성 실패"
            )
        )
    })?;
    let ckpt = format!("{ckpt_dir}/{model_name}");
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    valid_model
        .clone()
        .save_file(&ckpt, &recorder)
        .map_err(|e| {
            format!(
                "{}: {e}",
                tr("checkpoint save failed", "체크포인트 저장 실패")
            )
        })?;

    let report = TrainReport {
        model_name: model_name.to_string(),
        target: config.target.clone(),
        feature_names: feature_names.clone(),
        input_dim,
        output_dim,
        num_params,
        epochs: config.epochs,
        batch_size,
        learning_rate: lr as f32,
        final_train_loss,
        final_val_loss,
        predictions,
        targets: targets_out,
        checkpoint_path: format!("{ckpt}.json"),
        stopped_early,
        best_epoch,
    };

    Ok(RawTrained {
        model: valid_model,
        report,
        feature_names,
        fmean,
        fstd,
        target: config.target.clone(),
    })
}

/// Shared inference preprocessing: validates the frame, reads the feature columns
/// in training order, and applies the same standardization used during training.
/// Returns `(xs, rows, feature_count)`.
fn prepare_inference_input(
    trained: &TrainedModel,
    df: &DataFrame,
) -> Result<(Vec<f32>, usize, usize), String> {
    let feature_count = trained.feature_names.len();
    let n = df.height();
    if feature_count == 0 {
        return Err(tr(
            "The model has no feature information. Train it first with train().",
            "모델에 특성 정보가 없습니다. 먼저 train()으로 학습하세요.",
        )
        .into());
    }
    if n == 0 {
        return Err(tr("No data to predict on.", "예측할 데이터가 비어 있습니다.").into());
    }

    let mut col_vecs: Vec<Vec<f32>> = Vec::with_capacity(feature_count);
    for j in 0..feature_count {
        let name = &trained.feature_names[j];
        let col = df.column(name.as_str()).map_err(|e| {
            format!(
                "{} '{name}': {e}",
                tr("prediction feature column access failed", "예측 특성 컬럼")
            )
        })?;
        col_vecs.push(series_to_f32(col));
    }

    let mut xs = Vec::with_capacity(n * feature_count);
    for i in 0..n {
        for (j, col) in col_vecs.iter().enumerate().take(feature_count) {
            let v = col[i] as f64;
            if trained.model.raw_input {
                xs.push(if v.is_finite() { v as f32 } else { 0.0 });
            } else {
                let v = if v.is_finite() { v } else { trained.fmean[j] };
                xs.push(((v - trained.fmean[j]) / trained.fstd[j]) as f32);
            }
        }
    }
    if trained.model.raw_input {
        check_embedding_indices(&trained.layers, &xs, feature_count);
    }
    Ok((xs, n, feature_count))
}

/// Appends the prediction column to a clone of `df`.
fn attach_prediction(
    trained: &TrainedModel,
    df: &DataFrame,
    preds: &[f32],
    as_col: Option<&str>,
) -> Result<DataFrame, String> {
    let out_col = match as_col {
        Some(c) => c.to_string(),
        None => format!("{}_pred", trained.target),
    };
    let pred_f64: Vec<f64> = preds.iter().map(|&v| v as f64).collect();

    let mut out = df.clone();
    out.with_column(Column::new(out_col.into(), pred_f64))
        .map_err(|e| {
            format!(
                "{}: {e}",
                tr("prediction column add failed", "예측 컬럼 추가 실패")
            )
        })?;
    Ok(out)
}

/// `dataset |> predict(model_var, as: "col")` — adds a prediction column using the trained model.
///
/// Default prediction column name: `<target>_pred`. Runs on the CPU reference
/// backend using the in-memory model.
pub fn predict(
    trained: &TrainedModel,
    df: &DataFrame,
    as_col: Option<&str>,
) -> Result<DataFrame, String> {
    let (xs, n, feature_count) = prepare_inference_input(trained, df)?;

    let device: Device<Plain> = Default::default();
    let x = Tensor::<Plain, 2>::from_data(TensorData::new(xs, [n, feature_count]), &device);
    let mut infer_model = trained.model.clone();
    infer_model.training = false;
    let pred_t = infer_model.forward(x);
    let preds = pred_t.into_data().to_vec::<f32>().unwrap_or_default();

    attach_prediction(trained, df, &preds, as_col)
}

/// `predict` on an arbitrary Burn backend `B`.
///
/// Rebuilds the declared module graph on `B` and loads the portable checkpoint
/// saved during training, so a GPU provider runs the forward pass on its own
/// device. The checkpoint is backend-neutral, so the same artifact can be
/// evaluated on CPU and GPU with matching predictions.
pub fn predict_on<B>(
    trained: &TrainedModel,
    df: &DataFrame,
    as_col: Option<&str>,
) -> Result<DataFrame, String>
where
    B: Backend,
{
    let (xs, n, feature_count) = prepare_inference_input(trained, df)?;

    let device: Device<B> = Default::default();
    let template = build_mlp::<B>(&trained.layers, feature_count, &device)?;
    let recorder = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    let mut infer_model = template
        .load_file(trained.report.checkpoint_path.as_str(), &recorder, &device)
        .map_err(|e| {
            format!(
                "{}: {e}",
                tr("checkpoint load failed", "체크포인트 로드 실패")
            )
        })?;
    infer_model.training = false;

    let x = Tensor::<B, 2>::from_data(TensorData::new(xs, [n, feature_count]), &device);
    let pred_t = infer_model.forward(x);
    let preds = pred_t.into_data().to_vec::<f32>().unwrap_or_default();

    attach_prediction(trained, df, &preds, as_col)
}

/// Layered model registry helper: <model name, LayerKind list>.
pub type ModelRegistry = HashMap<String, Vec<LayerKind>>;

#[cfg(test)]
mod tests {
    use super::*;
    use xazz_compiler::ast::EmbeddingVocab;

    fn embedding_layer(vocab: EmbeddingVocab) -> LayerKind {
        LayerKind::Embedding {
            vocab,
            embed_dim: 2,
        }
    }

    #[test]
    fn leading_embedding_vocabs_expands_shared_and_per_column() {
        // A shared vocab is replicated for every input column.
        assert_eq!(
            leading_embedding_vocabs(&[embedding_layer(EmbeddingVocab::Shared(5))], 3),
            Some(vec![5, 5, 5])
        );
        // A per-column list is returned as-is when its length matches.
        assert_eq!(
            leading_embedding_vocabs(&[embedding_layer(EmbeddingVocab::PerColumn(vec![4, 7]))], 2),
            Some(vec![4, 7])
        );
        // A length mismatch cannot be diagnosed here (build_mlp rejects it).
        assert_eq!(
            leading_embedding_vocabs(&[embedding_layer(EmbeddingVocab::PerColumn(vec![4]))], 2),
            None
        );
        assert_eq!(leading_embedding_vocabs(&[LayerKind::Dense(4)], 2), None);
        assert_eq!(
            leading_embedding_vocabs(
                &[
                    LayerKind::Dense(4),
                    embedding_layer(EmbeddingVocab::Shared(5))
                ],
                2
            ),
            None,
            "Embedding 은 첫 레이어여야 한다"
        );
    }

    #[test]
    fn counts_out_of_range_per_column_against_each_vocab() {
        // 2 columns, row-major: col0 vocab 3, col1 vocab 5.
        let values = vec![0.0, 0.0, 3.0, 4.0, -1.0, 9.0];
        assert_eq!(count_out_of_range_per_column(&values, 2, &[3, 5]), 3);
        // A uniform shared vocab still checks every column.
        assert_eq!(
            count_out_of_range_per_column(&[0.0, 9.0, 2.0, 1.0], 2, &[3, 3]),
            1
        );
    }

    #[test]
    fn counts_only_out_of_range_finite_indices() {
        // [0, vocab-1] 안쪽은 세지 않는다.
        assert_eq!(count_out_of_range_indices(&[0.0, 1.0, 4.0], 5), 0);
        // vocab-1 초과
        assert_eq!(count_out_of_range_indices(&[0.0, 3.0, 4.0], 3), 2);
        // 음수 인덱스
        assert_eq!(count_out_of_range_indices(&[-1.0, 0.0], 5), 1);
        // non-finite 는 forward 에서 0 으로 매핑되므로 세지 않는다.
        assert_eq!(count_out_of_range_indices(&[f32::NAN, f32::INFINITY], 5), 0);
    }

    #[test]
    fn counts_only_non_integer_finite_indices() {
        // 정수값(소수부 0)은 세지 않는다.
        assert_eq!(count_non_integer_indices(&[0.0, 1.0, 4.0]), 0);
        // 소수부가 있는 연속형 값은 truncate 되므로 센다.
        assert_eq!(count_non_integer_indices(&[0.5, 1.0, 2.25]), 2);
        // 음수 비정수도 소수부가 버려진다.
        assert_eq!(count_non_integer_indices(&[-0.5, 3.0]), 1);
        // non-finite 는 forward 에서 0 으로 매핑되므로 세지 않는다.
        assert_eq!(count_non_integer_indices(&[f32::NAN, f32::INFINITY]), 0);
    }
}
