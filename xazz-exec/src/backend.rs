//! xazz-exec/src/backend.rs — ML compute-backend abstraction at the MLOp boundary.
//!
//! The Typed IR (`xazz_core::ir::MLOp`) is backend-independent. This module is the
//! single dispatch point between that IR and a concrete ML engine: `runtime`
//! calls [`active()`]`.train(..)` / `.predict(..)` instead of a concrete engine.
//!
//! Burn + burn-ndarray (pure-Rust CPU) is the first provider. The same trait is
//! the plug-in slot for the hardware-gated work:
//!
//!   - CUDA (`burn-tch`) and WebGPU (`burn-wgpu`) — issue D1 (#62)
//!   - ONNX Runtime interop — issue D2 (#63)
//!   - burn-engine / remote inference — issue F4 (#73)
//!
//! ## Selection contract
//!
//! `XAZZ_BACKEND` selects the provider: `cpu` (default) | `cuda` | `wgpu` | `onnx`.
//! A backend that was not compiled into the binary — or an unrecognised value —
//! falls back to CPU with an explicit warning; the runtime never silently
//! switches device without telling the operator.
//!
//! ```text
//! XAZZ_BACKEND=cuda xazz run model.xzz   # needs --features cuda + a CUDA host
//! ```
//!
//! CPU is always compiled: it is both the fallback and the reference
//! implementation for the parity acceptance tests. The GPU/ONNX acceptance
//! tests are `#[ignore]`d behind their feature (see the bottom of this file) so
//! they can be executed on a host that has the hardware — the implementation
//! lives behind the same trait, so enabling the feature is the only change.

use std::sync::OnceLock;

use polars::prelude::DataFrame;
use xazz_compiler::ast::{LayerKind, TrainConfig};
use xazz_core::i18n::{is_korean, tr};

use crate::dl::{SweepCombo, SweepReport, TrainedModel};

/// A pluggable ML engine behind the Typed IR's `MLOp` boundary.
///
/// `train` returns the serialized [`TrainedModel`] artifact (the ndarray
/// checkpoint format). That artifact is the backend-neutral handoff consumed by
/// `predict`, so a GPU provider can train on its device and materialise the
/// portable artifact for storage.
pub trait ComputeBackend: Send + Sync {
    /// Stable identifier surfaced in logs (`cpu`, `cuda`, `wgpu`, `onnx`).
    fn id(&self) -> &'static str;

    /// `dataset |> train(model, target: .., ..)`.
    fn train(
        &self,
        df: &DataFrame,
        model_name: &str,
        layers: &[LayerKind],
        config: &TrainConfig,
    ) -> Result<TrainedModel, String>;

    /// `dataset |> predict(model_var, as: "col")`.
    fn predict(
        &self,
        trained: &TrainedModel,
        df: &DataFrame,
        as_col: Option<&str>,
    ) -> Result<DataFrame, String>;

    /// `dataset |> train(model, ..)` over a hyperparameter grid (D3).
    ///
    /// The default implementation evaluates every combination via [`Self::train`]
    /// and returns the best model plus a per-combination report. Selection uses
    /// validation loss when a `validation_split` is configured, else training loss.
    fn sweep(
        &self,
        df: &DataFrame,
        model_name: &str,
        layers: &[LayerKind],
        config: &TrainConfig,
    ) -> Result<(TrainedModel, SweepReport), String> {
        let combos = config.expand_sweep();
        let mut best: Option<(usize, TrainedModel)> = None;
        let mut entries: Vec<SweepCombo> = Vec::with_capacity(combos.len());

        for (index, combo) in combos.iter().enumerate() {
            let trained = self.train(df, model_name, layers, combo)?;
            let report = &trained.report;
            let entry = SweepCombo {
                epochs: report.epochs,
                batch_size: report.batch_size,
                learning_rate: report.learning_rate,
                final_train_loss: report.final_train_loss,
                final_val_loss: report.final_val_loss,
                stopped_early: report.stopped_early,
                best_epoch: report.best_epoch,
                selected: false,
            };
            let is_better = match &best {
                None => true,
                Some((best_index, _)) => {
                    SweepReport::score(&entry) < SweepReport::score(&entries[*best_index])
                }
            };
            if is_better {
                best = Some((index, trained));
            }
            entries.push(entry);
        }

        let (best_index, best_model) = best.ok_or_else(|| {
            tr(
                "hyperparameter sweep produced no combinations.",
                "하이퍼파라미터 스윕 조합이 없습니다.",
            )
            .to_string()
        })?;
        entries[best_index].selected = true;

        // Every combination re-trains onto the same checkpoint path, so the file
        // left on disk is the last combination's. Re-save the winner so the
        // checkpoint matches the model returned for downstream predict().
        let base = best_model.report.checkpoint_path.trim_end_matches(".json");
        crate::dl::save_checkpoint(&best_model.model, base)?;

        let report = SweepReport {
            model_name: model_name.to_string(),
            target: config.target.clone(),
            combos: entries,
            best_index,
        };
        Ok((best_model, report))
    }
}

/// Reference provider: Burn + burn-ndarray (pure-Rust CPU).
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn id(&self) -> &'static str {
        BackendKind::Cpu.id()
    }

    fn train(
        &self,
        df: &DataFrame,
        model_name: &str,
        layers: &[LayerKind],
        config: &TrainConfig,
    ) -> Result<TrainedModel, String> {
        crate::dl::train(df, model_name, layers, config)
    }

    fn predict(
        &self,
        trained: &TrainedModel,
        df: &DataFrame,
        as_col: Option<&str>,
    ) -> Result<DataFrame, String> {
        crate::dl::predict(trained, df, as_col)
    }
}

/// The set of selectable providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Wgpu,
    Onnx,
}

impl BackendKind {
    /// Every kind the resolver understands, in selection order.
    pub const ALL: [BackendKind; 4] = [
        BackendKind::Cpu,
        BackendKind::Cuda,
        BackendKind::Wgpu,
        BackendKind::Onnx,
    ];

    /// Canonical id (also the `XAZZ_BACKEND` value).
    pub fn id(self) -> &'static str {
        match self {
            BackendKind::Cpu => "cpu",
            BackendKind::Cuda => "cuda",
            BackendKind::Wgpu => "wgpu",
            BackendKind::Onnx => "onnx",
        }
    }

    /// Parses an `XAZZ_BACKEND` value, accepting common engine aliases.
    /// Returns `None` for an unrecognised value.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" | "ndarray" | "burn" | "burn-ndarray" => Some(BackendKind::Cpu),
            "cuda" | "tch" | "torch" | "libtorch" | "burn-tch" => Some(BackendKind::Cuda),
            "wgpu" | "gpu" | "webgpu" | "burn-wgpu" => Some(BackendKind::Wgpu),
            "onnx" | "onnxruntime" | "ort" => Some(BackendKind::Onnx),
            _ => None,
        }
    }

    /// Whether this provider was compiled into the current binary.
    ///
    /// CPU is unconditional (fallback + reference). The others require the
    /// matching cargo feature. Device presence is probed by the provider once
    /// its real dependency is wired; unknown/absent device also falls back to
    /// CPU through the same warning path.
    pub fn is_compiled(self) -> bool {
        match self {
            BackendKind::Cpu => true,
            BackendKind::Cuda => cfg!(feature = "cuda"),
            BackendKind::Wgpu => cfg!(feature = "wgpu"),
            BackendKind::Onnx => cfg!(feature = "onnx"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature-gated provider scaffolds
//
// Each is the place a real provider lands: swap the body for the burn-tch /
// burn-wgpu / onnxruntime implementation. The acceptance test beside the trait
// pins the contract those providers must satisfy.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use xazz_core::i18n::tr;

    /// CUDA provider (`burn-tch`). Scaffold — see issue D1 (#62).
    pub struct CudaBackend;

    impl ComputeBackend for CudaBackend {
        fn id(&self) -> &'static str {
            BackendKind::Cuda.id()
        }

        fn train(
            &self,
            _df: &DataFrame,
            _model_name: &str,
            _layers: &[LayerKind],
            _config: &TrainConfig,
        ) -> Result<TrainedModel, String> {
            Err(tr(
                "CUDA backend is a scaffold: add `burn-tch` and implement it (issue #62).",
                "CUDA 백엔드는 스캐폴드입니다: `burn-tch`를 추가하고 구현하세요 (이슈 #62).",
            )
            .into())
        }

        fn predict(
            &self,
            _trained: &TrainedModel,
            _df: &DataFrame,
            _as_col: Option<&str>,
        ) -> Result<DataFrame, String> {
            Err(tr(
                "CUDA backend is a scaffold: add `burn-tch` and implement it (issue #62).",
                "CUDA 백엔드는 스캐폴드입니다: `burn-tch`를 추가하고 구현하세요 (이슈 #62).",
            )
            .into())
        }
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn_wgpu::Wgpu;

    /// WebGPU provider (`burn-wgpu`). Trains and predicts on a Vulkan/Metal/DX12
    /// device; the portable checkpoint round-trips back to CPU for storage.
    pub struct WgpuBackend;

    impl ComputeBackend for WgpuBackend {
        fn id(&self) -> &'static str {
            BackendKind::Wgpu.id()
        }

        fn train(
            &self,
            df: &DataFrame,
            model_name: &str,
            layers: &[LayerKind],
            config: &TrainConfig,
        ) -> Result<TrainedModel, String> {
            crate::dl::train_on::<Wgpu<f32>>(df, model_name, layers, config)
        }

        fn predict(
            &self,
            trained: &TrainedModel,
            df: &DataFrame,
            as_col: Option<&str>,
        ) -> Result<DataFrame, String> {
            crate::dl::predict_on::<Wgpu<f32>>(trained, df, as_col)
        }
    }
}

#[cfg(feature = "onnx")]
mod onnx {
    use super::*;
    use xazz_core::i18n::tr;

    /// ONNX Runtime provider. Scaffold — see issue D2 (#63).
    pub struct OnnxBackend;

    impl ComputeBackend for OnnxBackend {
        fn id(&self) -> &'static str {
            BackendKind::Onnx.id()
        }

        fn train(
            &self,
            _df: &DataFrame,
            _model_name: &str,
            _layers: &[LayerKind],
            _config: &TrainConfig,
        ) -> Result<TrainedModel, String> {
            Err(tr(
                "ONNX backend is a scaffold: add `onnxruntime` and implement it (issue #63).",
                "ONNX 백엔드는 스캐폴드입니다: `onnxruntime`을 추가하고 구현하세요 (이슈 #63).",
            )
            .into())
        }

        fn predict(
            &self,
            _trained: &TrainedModel,
            _df: &DataFrame,
            _as_col: Option<&str>,
        ) -> Result<DataFrame, String> {
            Err(tr(
                "ONNX backend is a scaffold: add `onnxruntime` and implement it (issue #63).",
                "ONNX 백엔드는 스캐폴드입니다: `onnxruntime`을 추가하고 구현하세요 (이슈 #63).",
            )
            .into())
        }
    }
}

/// Instantiates a provider that the resolver has already checked is compiled.
fn build(kind: BackendKind) -> Box<dyn ComputeBackend> {
    match kind {
        BackendKind::Cpu => Box::new(CpuBackend),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Box::new(cuda::CudaBackend),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => Box::new(wgpu::WgpuBackend),
        #[cfg(feature = "onnx")]
        BackendKind::Onnx => Box::new(onnx::OnnxBackend),
        #[allow(unreachable_patterns)]
        _ => Box::new(CpuBackend),
    }
}

fn fallback_warning(kind: BackendKind) -> String {
    if is_korean() {
        format!(
            "XAZZ_BACKEND={} 백엔드가 이 바이너리에 포함되지 않았습니다. CPU로 폴백합니다.",
            kind.id()
        )
    } else {
        format!(
            "XAZZ_BACKEND={} is not compiled into this binary; falling back to CPU.",
            kind.id()
        )
    }
}

fn unknown_warning(raw: &str) -> String {
    if is_korean() {
        format!(
            "알 수 없는 XAZZ_BACKEND='{raw}' 입니다. 사용 가능: {}. CPU로 폴백합니다.",
            BackendKind::ALL
                .iter()
                .map(|k| k.id())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        format!(
            "unknown XAZZ_BACKEND='{raw}'. Available: {}. Falling back to CPU.",
            BackendKind::ALL
                .iter()
                .map(|k| k.id())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Resolves a requested backend value into a provider.
///
/// Pure (no environment access) so it is directly testable. Returns the provider
/// plus an optional warning describing a fallback; `None` means the request was
/// honoured (or absent → CPU).
pub fn resolve(requested: Option<&str>) -> (Box<dyn ComputeBackend>, Option<String>) {
    match requested {
        None => (build(BackendKind::Cpu), None),
        Some(raw) => match BackendKind::parse(raw) {
            Some(kind) if kind.is_compiled() => (build(kind), None),
            Some(kind) => (build(BackendKind::Cpu), Some(fallback_warning(kind))),
            None => (build(BackendKind::Cpu), Some(unknown_warning(raw))),
        },
    }
}

/// The process-wide active provider, chosen once from `XAZZ_BACKEND`.
///
/// On first use it logs the fallback warning (if any) or the selected non-CPU
/// backend to stderr. Subsequent calls reuse the resolved provider.
pub fn active() -> &'static dyn ComputeBackend {
    static ACTIVE: OnceLock<Box<dyn ComputeBackend>> = OnceLock::new();
    ACTIVE
        .get_or_init(|| {
            let requested = std::env::var("XAZZ_BACKEND").ok();
            let (backend, warning) = resolve(requested.as_deref());
            match warning {
                Some(msg) => eprintln!("[xazz] {msg}"),
                None if backend.id() != BackendKind::Cpu.id() => {
                    eprintln!("[xazz] ML backend: {}", backend.id())
                }
                None => {}
            }
            backend
        })
        .as_ref()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny separable regression set shared by the CPU round-trip test and the
    /// hardware acceptance tests.
    #[cfg(any(test, feature = "cuda", feature = "wgpu", feature = "onnx"))]
    pub(super) fn tiny_dataset() -> (DataFrame, Vec<LayerKind>, TrainConfig) {
        use polars::prelude::*;
        let df = df!(
            "x1" => [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
            "x2" => [1.0f64, 1.0, 0.0, 0.0, 1.0, 1.0],
            "y"  => [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .expect("tiny dataset");
        let layers = vec![LayerKind::Dense(4), LayerKind::ReLU, LayerKind::Dense(1)];
        let config = TrainConfig {
            target: "y".to_string(),
            epochs: 2,
            learning_rate: 0.05,
            batch_size: Some(3),
            validation_split: None,
            early_stopping_patience: None,
            sweep: Default::default(),
        };
        (df, layers, config)
    }

    fn cleanup(checkpoint_path: &str) {
        let _ = std::fs::remove_file(checkpoint_path);
        let _ = std::fs::remove_dir("checkpoints");
    }

    #[test]
    fn parse_accepts_aliases_and_rejects_garbage() {
        assert_eq!(BackendKind::parse("cpu"), Some(BackendKind::Cpu));
        assert_eq!(BackendKind::parse(""), Some(BackendKind::Cpu));
        assert_eq!(BackendKind::parse(" burn-ndarray "), Some(BackendKind::Cpu));
        assert_eq!(BackendKind::parse("CUDA"), Some(BackendKind::Cuda));
        assert_eq!(BackendKind::parse("tch"), Some(BackendKind::Cuda));
        assert_eq!(BackendKind::parse("webgpu"), Some(BackendKind::Wgpu));
        assert_eq!(BackendKind::parse("ort"), Some(BackendKind::Onnx));
        assert_eq!(BackendKind::parse("quantum"), None);
    }

    #[test]
    fn resolve_defaults_to_cpu_without_warning() {
        let (backend, warning) = resolve(None);
        assert_eq!(backend.id(), "cpu");
        assert!(warning.is_none());
    }

    #[test]
    fn resolve_unknown_value_falls_back_with_warning() {
        let (backend, warning) = resolve(Some("quantum"));
        assert_eq!(backend.id(), "cpu");
        assert!(warning.unwrap().contains("quantum"));
    }

    #[test]
    fn resolve_uncompiled_backend_falls_back_with_warning() {
        let (backend, warning) = resolve(Some("cuda"));
        if cfg!(feature = "cuda") {
            assert_eq!(backend.id(), "cuda");
            assert!(warning.is_none());
        } else {
            assert_eq!(backend.id(), "cpu");
            assert!(warning.unwrap().contains("cuda"));
        }
    }

    #[test]
    fn cpu_backend_trains_and_predicts_through_the_trait() {
        let (df, layers, config) = tiny_dataset();
        let (backend, warning) = resolve(None);
        assert!(warning.is_none());

        let trained = backend
            .train(&df, "backend_unit_mlp", &layers, &config)
            .expect("cpu train");
        assert_eq!(trained.report.input_dim, 2);
        assert_eq!(trained.report.output_dim, 1);

        let out = backend
            .predict(&trained, &df, Some("pred"))
            .expect("cpu predict");
        assert_eq!(out.height(), df.height());
        assert!(out.column("pred").is_ok());

        cleanup(&trained.report.checkpoint_path);
    }

    /// D3 CNN: a Conv1d -> ReLU -> Dense model trains and predicts end-to-end on CPU.
    #[test]
    fn cpu_backend_trains_conv1d_model() {
        use polars::prelude::*;

        let df = df!(
            "x1" => [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "x2" => [1.0f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "x3" => [0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "x4" => [2.0f64, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
            "y"  => [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )
        .expect("conv dataset");

        let layers = vec![
            LayerKind::Conv1d {
                out_channels: 4,
                kernel_size: 3,
            },
            LayerKind::ReLU,
            LayerKind::Dense(1),
        ];
        let config = TrainConfig {
            target: "y".to_string(),
            epochs: 2,
            learning_rate: 0.05,
            batch_size: Some(4),
            validation_split: None,
            early_stopping_patience: None,
            sweep: Default::default(),
        };

        let (backend, warning) = resolve(None);
        assert!(warning.is_none());

        let trained = backend
            .train(&df, "backend_unit_conv1d", &layers, &config)
            .expect("cpu conv1d train");
        assert_eq!(trained.report.input_dim, 4);
        assert_eq!(trained.report.output_dim, 1);

        let out = backend
            .predict(&trained, &df, Some("pred"))
            .expect("cpu conv1d predict");
        assert_eq!(out.height(), df.height());
        assert!(out.column("pred").is_ok());

        cleanup(&trained.report.checkpoint_path);
    }

    /// D3: a model whose final output is not a single scalar (e.g. Conv1d without a
    /// final Dense(1)) is rejected instead of silently broadcasting the target.
    #[test]
    fn cpu_backend_rejects_non_scalar_output() {
        let (df, _layers, config) = tiny_dataset();
        let layers = vec![
            LayerKind::Conv1d {
                out_channels: 4,
                kernel_size: 3,
            },
            LayerKind::ReLU,
        ];

        let (backend, warning) = resolve(None);
        assert!(warning.is_none());

        let err = backend
            .train(&df, "backend_unit_bad_dim", &layers, &config)
            .expect_err("multi-output model must be rejected");
        assert!(err.contains("Dense(1)"), "오류 메시지에 안내가 없음: {err}");
    }

    /// D3 Embedding: an Embedding -> ReLU -> Dense model trains and predicts end-to-end on CPU.
    #[test]
    fn cpu_backend_trains_embedding_model() {
        use polars::prelude::*;

        let df = df!(
            "cat1" => [0i64, 1, 2, 3, 0, 1, 2, 3],
            "cat2" => [1i64, 0, 1, 0, 1, 0, 1, 0],
            "y"    => [0.0f64, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
        )
        .expect("embedding dataset");

        let layers = vec![
            LayerKind::Embedding {
                vocab_size: 4,
                embed_dim: 3,
            },
            LayerKind::ReLU,
            LayerKind::Dense(1),
        ];
        let config = TrainConfig {
            target: "y".to_string(),
            epochs: 3,
            learning_rate: 0.05,
            batch_size: Some(4),
            validation_split: None,
            early_stopping_patience: None,
            sweep: Default::default(),
        };

        let (backend, warning) = resolve(None);
        assert!(warning.is_none());

        let trained = backend
            .train(&df, "backend_unit_embedding", &layers, &config)
            .expect("cpu embedding train");
        assert_eq!(trained.report.input_dim, 2);
        assert_eq!(trained.report.output_dim, 1);

        let out = backend
            .predict(&trained, &df, Some("pred"))
            .expect("cpu embedding predict");
        assert_eq!(out.height(), df.height());
        assert!(out.column("pred").is_ok());

        cleanup(&trained.report.checkpoint_path);
    }

    /// D3 sweep: the grid is fully evaluated and the selected best model is usable.
    #[test]
    fn cpu_backend_sweep_selects_best_combination() {
        let (df, layers, mut config) = tiny_dataset();
        config.sweep.epochs = vec![2, 3];
        config.sweep.learning_rate = vec![0.05, 0.01];
        assert!(config.is_sweep());

        let (backend, warning) = resolve(None);
        assert!(warning.is_none());

        let (trained, report) = backend
            .sweep(&df, "backend_unit_sweep", &layers, &config)
            .expect("cpu sweep");
        assert_eq!(report.combos.len(), 4, "2×2 그리드");
        assert!(
            report.combos[report.best_index].selected,
            "best_index 조합이 선택되어야 함"
        );
        assert_eq!(
            report.combos.iter().filter(|c| c.selected).count(),
            1,
            "선택 조합은 하나여야 함"
        );
        assert!(
            report.combos[report.best_index]
                .final_train_loss
                .is_finite(),
            "최적 조합 손실이 유한해야 함"
        );

        let out = backend
            .predict(&trained, &df, Some("pred"))
            .expect("cpu sweep predict");
        assert_eq!(out.height(), df.height());
        assert!(out.column("pred").is_ok());

        cleanup(&trained.report.checkpoint_path);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hardware acceptance tests (issue D1 #62, D2 #63)
//
// These require a device/SDK absent in CI, so they are `#[ignore]`d; on a
// matching host run:
//
//   cargo test -p xazz-exec --features wgpu   -- --ignored
//   cargo test -p xazz-exec --features cuda   -- --ignored
//   cargo test -p xazz-exec --features onnx   -- --ignored
//
// `wgpu` is implemented: it trains on the device and evaluates the portable
// checkpoint on the same device. The parity check compares inference from one
// shared CPU-trained checkpoint on CPU vs the requested backend (identical
// weights), rather than two independently-initialised training runs — random
// initialisers differ across backends, so comparing losses from separate runs
// would not be meaningful.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, any(feature = "cuda", feature = "wgpu", feature = "onnx")))]
mod acceptance {
    use super::ComputeBackend;
    use super::tests::tiny_dataset;
    use polars::prelude::*;

    fn predictions(df: &DataFrame) -> Vec<f64> {
        df.column("pred")
            .expect("prediction column")
            .f64()
            .expect("float prediction column")
            .into_no_null_iter()
            .collect()
    }

    /// Trains the requested backend for real, then verifies numerical parity by
    /// evaluating the same CPU-trained checkpoint on CPU and on `requested`.
    #[cfg(any(feature = "cuda", feature = "wgpu", feature = "onnx"))]
    fn assert_parity(requested: &str) {
        let (df, layers, config) = tiny_dataset();
        let cpu = super::CpuBackend
            .train(&df, "acc_cpu", &layers, &config)
            .expect("cpu reference train");

        let (backend, warning) = super::resolve(Some(requested));
        assert!(warning.is_none(), "{requested} should be compiled");
        assert_eq!(backend.id(), requested);

        // Real device training must complete and produce a finite loss.
        let gpu = backend
            .train(&df, "acc_gpu", &layers, &config)
            .expect("backend train");
        assert!(
            gpu.report.final_train_loss.is_finite(),
            "{requested} produced a non-finite training loss"
        );

        // Numerical parity: same weights (cpu checkpoint) evaluated on CPU and device.
        let cpu_out = super::CpuBackend
            .predict(&cpu, &df, Some("pred"))
            .expect("cpu predict");
        let gpu_out = backend
            .predict(&cpu, &df, Some("pred"))
            .expect("backend predict");
        let max_diff = predictions(&cpu_out)
            .into_iter()
            .zip(predictions(&gpu_out))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-3,
            "{requested} inference diverged from CPU by {max_diff}"
        );

        // The device-trained artifact is usable for inference on the device.
        let out = backend
            .predict(&gpu, &df, Some("pred"))
            .expect("backend predict (device-trained)");
        assert_eq!(out.height(), df.height());

        let _ = std::fs::remove_file(&cpu.report.checkpoint_path);
        let _ = std::fs::remove_file(&gpu.report.checkpoint_path);
        let _ = std::fs::remove_dir("checkpoints");
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU + burn-tch: `cargo test -p xazz-exec --features cuda -- --ignored` (issue #62)"]
    fn cuda_matches_cpu_losses() {
        assert_parity("cuda");
    }

    #[cfg(feature = "wgpu")]
    #[test]
    #[ignore = "requires a WebGPU device + burn-wgpu: `cargo test -p xazz-exec --features wgpu -- --ignored` (issue #62)"]
    fn wgpu_matches_cpu_losses() {
        assert_parity("wgpu");
    }

    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "requires onnxruntime + an exported model: `cargo test -p xazz-exec --features onnx -- --ignored` (issue #63)"]
    fn onnx_matches_cpu_losses() {
        assert_parity("onnx");
    }
}
