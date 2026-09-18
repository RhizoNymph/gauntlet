//! Node-side mode. Invoked over ssh by the orchestrator; never run by hand.
//!
//! stdout carries the JSON-lines event protocol (see `crate::proto`), so
//! nothing in agent mode may print to stdout except through `EventSink`.
//! Logs go to stderr via `tracing`.

pub mod barrier;
pub mod counters;
pub mod cpu;
pub mod disk;
pub mod inventory;
pub mod mem;
pub mod nccl;
pub mod net;
pub mod window;

#[cfg(feature = "gpu")]
pub mod gpu;

use std::io::{BufRead, Write};
use std::sync::Mutex;

use anyhow::{Context, Result};

use crate::cli::AgentRunArgs;
use crate::proto::{
    AgentEvent, AgentTaskSpec, LogLevel, MetricRecord, PROTO_VERSION, Phase, Scope, TestId,
    TestOutcome,
};

/// Serialized writer for protocol events. Cloneable-by-reference across
/// worker threads; one event per line, flushed immediately so the
/// orchestrator sees progress live.
pub struct EventSink {
    out: Mutex<Box<dyn Write + Send>>,
}

impl EventSink {
    pub fn stdout() -> Self {
        Self {
            out: Mutex::new(Box::new(std::io::stdout())),
        }
    }

    /// Sink writing to an arbitrary writer (tests).
    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            out: Mutex::new(writer),
        }
    }

    pub fn emit(&self, event: &AgentEvent) {
        let line = crate::proto::encode_event(event);
        let mut out = self.out.lock().expect("event sink poisoned");
        // An unwritable stdout means the ssh channel is gone; nothing useful
        // can be done from the agent side, so exit hard.
        if writeln!(out, "{line}").and_then(|_| out.flush()).is_err() {
            std::process::exit(3);
        }
    }

    pub fn metric(&self, record: MetricRecord) {
        self.emit(&AgentEvent::Metric { record });
    }

    pub fn outcome(&self, test: TestId, scope: Scope, outcome: TestOutcome) {
        self.emit(&AgentEvent::Outcome {
            test,
            scope,
            outcome,
        });
    }

    pub fn log(&self, level: LogLevel, message: impl Into<String>) {
        self.emit(&AgentEvent::Log {
            level,
            message: message.into(),
        });
    }
}

/// Entry point for `gauntlet agent run`: reads an `AgentTaskSpec` JSON
/// document from stdin, then executes the requested phases in order.
pub async fn run(args: AgentRunArgs) -> Result<()> {
    let mut input = String::new();
    std::io::stdin()
        .lock()
        .read_line(&mut input)
        .context("reading task spec from stdin")?;
    let mut spec: AgentTaskSpec =
        serde_json::from_str(input.trim()).context("parsing AgentTaskSpec")?;
    if !args.phases.is_empty() {
        spec.phases = args
            .phases
            .iter()
            .map(|name| Phase::parse(name).with_context(|| format!("unknown phase {name:?}")))
            .collect::<Result<Vec<_>>>()?;
    }

    let sink = EventSink::stdout();
    sink.emit(&AgentEvent::Hello {
        proto_version: PROTO_VERSION,
        hostname: hostname()?,
    });

    for phase in &spec.phases {
        sink.emit(&AgentEvent::PhaseStart { phase: *phase });
        let result = match phase {
            Phase::Inventory => inventory::run(&sink),
            Phase::CpuMem => cpu_mem_phase(&sink, &spec),
            Phase::Gpu => gpu_phase(&sink, &spec),
            // Network tests are driven pairwise by the orchestrator through
            // `agent peer` / `agent nccl`, not from the phase loop.
            Phase::Network => Ok(()),
            Phase::Overlap => overlap_phase(&sink, &spec),
        };
        if let Err(error) = result {
            sink.emit(&AgentEvent::Fatal {
                message: format!("{phase:?} phase failed: {error:#}"),
            });
            return Err(error);
        }
        sink.emit(&AgentEvent::PhaseEnd { phase: *phase });
    }
    // Counter passes run after any listed phases; the orchestrator sends
    // them as dedicated invocations with an empty phase list.
    if let Some(request) = &spec.counters {
        counters::run(&sink, request);
    }
    Ok(())
}

fn cpu_mem_phase(sink: &EventSink, spec: &AgentTaskSpec) -> Result<()> {
    cpu::run(sink, &spec.cpu)?;
    mem::run(sink, &spec.mem)?;
    disk::run(sink, &spec.disk)?;
    Ok(())
}

#[cfg(feature = "gpu")]
fn gpu_phase(sink: &EventSink, spec: &AgentTaskSpec) -> Result<()> {
    gpu::run(sink, &spec.gpu)
}

#[cfg(not(feature = "gpu"))]
fn gpu_phase(sink: &EventSink, _spec: &AgentTaskSpec) -> Result<()> {
    sink.outcome(
        TestId::GpuGemmCorrectness,
        Scope::Node,
        TestOutcome::Skipped {
            reason: "agent built without gpu feature".into(),
        },
    );
    Ok(())
}

#[cfg(feature = "gpu")]
fn overlap_phase(sink: &EventSink, spec: &AgentTaskSpec) -> Result<()> {
    gpu::overlap::run(sink, &spec.overlap)
}

#[cfg(not(feature = "gpu"))]
fn overlap_phase(sink: &EventSink, _spec: &AgentTaskSpec) -> Result<()> {
    sink.outcome(
        TestId::OverlapGemm,
        Scope::Node,
        TestOutcome::Skipped {
            reason: "agent built without gpu feature".into(),
        },
    );
    Ok(())
}

/// Entry point for `gauntlet agent probe`: print an `InventorySnapshot`
/// JSON document on stdout (used by bootstrap for capability detection).
pub fn probe() -> Result<()> {
    let snapshot = inventory::collect()?;
    println!("{}", serde_json::to_string_pretty(&snapshot)?);
    Ok(())
}

pub fn hostname() -> Result<String> {
    let name = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .context("reading /proc/sys/kernel/hostname")?;
    Ok(name.trim().to_string())
}
