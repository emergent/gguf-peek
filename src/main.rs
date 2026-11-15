mod gguf;

use std::collections::{BTreeMap, BTreeSet};
use std::process::ExitCode;

use gguf::{GgmlType, Gguf, TensorInfo};

/// llama.cpp `llama_tensor_get_type_impl()` in src/llama-quant.cpp.
/// Short-circuiting keeps `i - n/8` from underflowing.
fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
}

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gguf-peek <model.gguf>");
        return ExitCode::FAILURE;
    };

    let g = match gguf::read(&path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    print_header(&path, &g);
    print_metadata(&g);
    print_type_totals(&g);
    print_kind_matrix(&g);
    print_more_bits_check(&g);

    ExitCode::SUCCESS
}

fn print_header(path: &str, g: &Gguf) {
    println!("== {path}");
    println!("GGUF version      : {}", g.version);
    println!("architecture      : {}", g.architecture().unwrap_or("?"));
    println!("tensors           : {}", g.tensors.len());
    println!("metadata entries  : {}", g.metadata.len());
    println!("alignment         : {}", g.alignment());
    println!(
        "tensor data offset: {} (0x{:x})",
        g.tensor_data_offset, g.tensor_data_offset
    );
    match g.block_count() {
        Some(n) => println!("block_count       : {n}"),
        None => println!("block_count       : (not found)"),
    }
    println!();
}

fn print_metadata(g: &Gguf) {
    println!("== metadata");
    for (k, v) in &g.metadata {
        println!("  {k:<44} {}", v.summary());
    }
    println!();
}

fn print_type_totals(g: &Gguf) {
    let mut by_type: BTreeMap<GgmlType, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let e = by_type.entry(t.ty).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.elements();
    }

    let total_elems: u64 = by_type.values().map(|(_, e)| e).sum();

    println!("== tensor types");
    println!(
        "  {:<10} {:>7} {:>18} {:>8}",
        "type", "count", "elements", "share"
    );
    for (ty, (count, elems)) in &by_type {
        let share = 100.0 * *elems as f64 / total_elems as f64;
        println!("  {:<10} {count:>7} {elems:>18} {share:>7.2}%", ty.name());
    }
    println!();
}

/// Groups `blk.N.<kind>` tensors by `<kind>`, so the per-layer type mixture shows up.
fn print_kind_matrix(g: &Gguf) {
    let mut kinds: BTreeMap<String, BTreeMap<GgmlType, usize>> = BTreeMap::new();
    let mut singles: Vec<&TensorInfo> = Vec::new();

    for t in &g.tensors {
        match t.layer() {
            Some((_, kind)) => {
                *kinds
                    .entry(kind.to_string())
                    .or_default()
                    .entry(t.ty)
                    .or_insert(0) += 1;
            }
            None => singles.push(t),
        }
    }

    println!("== per-layer tensors (blk.N.*)");
    for (kind, types) in &kinds {
        let mixed = if types.len() > 1 { "   <- mixed" } else { "" };
        let detail: Vec<String> = types
            .iter()
            .map(|(ty, n)| format!("{}x{}", ty.name(), n))
            .collect();
        println!("  {:<28} {}{}", kind, detail.join(", "), mixed);
    }
    println!();

    println!("== standalone tensors");
    for t in singles {
        println!("  {:<28} {:<8} {:?}", t.name, t.ty.name(), t.dims);
    }
    println!();
}

/// The payoff: predict promoted layers from the formula, then check against the file.
fn print_more_bits_check(g: &Gguf) {
    let Some(n) = g.block_count().map(|n| n as usize) else {
        println!("== use_more_bits check: skipped (block_count unknown)");
        return;
    };

    let mut by_kind: BTreeMap<&str, BTreeMap<usize, GgmlType>> = BTreeMap::new();
    for t in &g.tensors {
        if let Some((layer, kind)) = t.layer() {
            by_kind.entry(kind).or_default().insert(layer, t.ty);
        }
    }

    println!("== use_more_bits(i, {n}) vs actual");
    for (kind, layers) in &by_kind {
        let distinct: BTreeSet<GgmlType> = layers.values().copied().collect();
        if distinct.len() < 2 {
            continue;
        }

        let base = *distinct.iter().next().unwrap();
        let promoted = *distinct.iter().next_back().unwrap();

        let mut agree = 0usize;
        let mut disagree: Vec<usize> = Vec::new();
        for (&i, &ty) in layers {
            let predicted = if use_more_bits(i, n) { promoted } else { base };
            if predicted == ty {
                agree += 1;
            } else {
                disagree.push(i);
            }
        }

        println!(
            "  {:<28} base={} promoted={} -> {}/{} layers match",
            kind,
            base.name(),
            promoted.name(),
            agree,
            layers.len()
        );
        if !disagree.is_empty() {
            let shown: Vec<String> = disagree.iter().take(16).map(|i| i.to_string()).collect();
            println!("    mismatched layers: {}", shown.join(", "));
        }

        let actual: String = (0..n)
            .map(|i| match layers.get(&i) {
                Some(&ty) if ty == promoted => '#',
                Some(_) => '.',
                None => ' ',
            })
            .collect();
        let predicted: String = (0..n)
            .map(|i| if use_more_bits(i, n) { '#' } else { '.' })
            .collect();
        println!("    by block index");
        println!("      actual   : {actual}");
        println!("      predicted: {predicted}");
    }
    println!();
}
