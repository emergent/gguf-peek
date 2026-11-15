use std::io;

/// IEEE 754 binary16 -> f32 (std has no stable f16 yet).
///
/// f16: sign 1 bit | exponent 5 bits (bias 15)  | mantissa 10 bits
/// f32: sign 1 bit | exponent 8 bits (bias 127) | mantissa 23 bits
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    match exp {
        // zero and subnormals: mant * 2^-24, which f32 holds exactly
        0 => {
            const TWO_POW_NEG_24: f32 = 1.0 / 16_777_216.0;
            let magnitude = mant as f32 * TWO_POW_NEG_24;
            f32::from_bits(sign | magnitude.to_bits())
        }
        // infinity (mant == 0) and NaN (mant != 0)
        0x1F => f32::from_bits(sign | (0xFF << 23) | (mant << 13)),
        // normal numbers: rebias the exponent, left-align the mantissa
        _ => {
            let exp = exp + 127 - 15;
            f32::from_bits(sign | (exp << 23) | (mant << 13))
        }
    }
}

/// Splits `data` into blocks, refusing a trailing partial block.
fn whole_blocks<'a>(
    data: &'a [u8],
    block_bytes: usize,
    ty: &str,
) -> io::Result<std::slice::ChunksExact<'a, u8>> {
    if data.len() % block_bytes != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{ty} data must be a multiple of {block_bytes} bytes, got {}",
                data.len()
            ),
        ));
    }
    Ok(data.chunks_exact(block_bytes))
}

/// Elements per Q4_0 block.
pub const QK4_0: usize = 32;
/// Bytes per Q4_0 block: f16 scale (2) + 32 nibbles (16).
pub const Q4_0_BLOCK_BYTES: usize = 2 + QK4_0 / 2;

/// Q4_0: y = (q - 8) * d, with q a 4-bit value.
///
/// The nibbles are NOT stored as adjacent pairs. Byte j holds element j in its
/// low nibble and element j + 16 in its high nibble.
pub fn dequantize_q4_0(data: &[u8]) -> io::Result<Vec<f32>> {
    let blocks = whole_blocks(data, Q4_0_BLOCK_BYTES, "Q4_0")?;
    let mut out = Vec::with_capacity(blocks.len() * QK4_0);
    for block in blocks {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];

        let mut y = [0.0f32; QK4_0];
        for (j, &byte) in qs.iter().enumerate() {
            y[j] = ((byte & 0x0F) as i32 - 8) as f32 * d;
            y[j + QK4_0 / 2] = ((byte >> 4) as i32 - 8) as f32 * d;
        }
        out.extend_from_slice(&y);
    }
    Ok(out)
}

/// Elements per k-quants super-block.
pub const QK_K: usize = 256;

/// Q4_K packs 8 six-bit scales and 8 six-bit mins into 12 bytes.
/// Returns (scale, min) of sub-block `j` (0..8).
///
/// ```text
/// byte 0..4 : [s4..s7 high 2 bits | s0..s3 (6 bits)]
/// byte 4..8 : [m4..m7 high 2 bits | m0..m3 (6 bits)]
/// byte 8..12: [m4..m7 low 4 bits  | s4..s7 low 4 bits]
/// ```
pub fn scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Bytes per Q4_K super-block: d (2) + dmin (2) + scales (12) + 256 nibbles (128).
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;

/// Q4_K: y = d * scale * q - dmin * min, per 32-element sub-block.
pub fn dequantize_q4_k(data: &[u8]) -> io::Result<Vec<f32>> {
    let blocks = whole_blocks(data, Q4_K_BLOCK_BYTES, "Q4_K")?;
    let mut out = Vec::with_capacity(blocks.len() * QK_K);
    for block in blocks {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales: &[u8; 12] = block[4..16].try_into().unwrap();
        let qs = &block[16..];

        // 32 bytes of qs feed 64 values: low nibbles to sub-block 2i,
        // high nibbles to sub-block 2i + 1.
        for (i, q) in qs.chunks_exact(32).enumerate() {
            let (sc, m) = scale_min_k4(2 * i, scales);
            let (d1, m1) = (d * sc as f32, dmin * m as f32);
            let (sc, m) = scale_min_k4(2 * i + 1, scales);
            let (d2, m2) = (d * sc as f32, dmin * m as f32);

            out.extend(q.iter().map(|&b| d1 * (b & 0x0F) as f32 - m1));
            out.extend(q.iter().map(|&b| d2 * (b >> 4) as f32 - m2));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const F16_ONE: [u8; 2] = 0x3C00u16.to_le_bytes();

    /// One Q4_0 block: f16 scale `d` followed by 16 bytes of packed nibbles.
    fn q4_0_block(d: [u8; 2], qs: [u8; 16]) -> Vec<u8> {
        [d.as_slice(), qs.as_slice()].concat()
    }

    #[test]
    fn q4_0_nibble_8_is_zero() {
        let y = dequantize_q4_0(&q4_0_block(F16_ONE, [0x88; 16])).unwrap();
        assert_eq!(y, vec![0.0; 32]);
    }

    #[test]
    fn q4_0_low_nibbles_are_first_half_high_nibbles_second_half() {
        let mut qs = [0x88; 16];
        qs[0] = 0xF0; // low nibble 0 -> -8, high nibble 15 -> +7
        qs[1] = 0x19; // low nibble 9 -> +1, high nibble 1 -> -7
        let y = dequantize_q4_0(&q4_0_block(F16_ONE, qs)).unwrap();

        assert_eq!(y[0], -8.0);
        assert_eq!(y[1], 1.0);
        assert_eq!(y[16], 7.0);
        assert_eq!(y[17], -7.0);
        assert!(
            y.iter()
                .enumerate()
                .all(|(i, &v)| [0, 1, 16, 17].contains(&i) || v == 0.0)
        );
    }

    #[test]
    fn q4_0_scales_by_d() {
        let half = 0x3800u16.to_le_bytes(); // 0.5
        let y = dequantize_q4_0(&q4_0_block(half, [0x99; 16])).unwrap();
        assert_eq!(y, vec![0.5; 32]);
    }

    #[test]
    fn q4_0_multiple_blocks_are_concatenated() {
        let two = 0x4000u16.to_le_bytes(); // 2.0
        let data = [q4_0_block(F16_ONE, [0x99; 16]), q4_0_block(two, [0x77; 16])].concat();
        let y = dequantize_q4_0(&data).unwrap();

        assert_eq!(y.len(), 64);
        assert_eq!(&y[..32], &[1.0; 32]);
        assert_eq!(&y[32..], &[-2.0; 32]);
    }

    #[test]
    fn q4_0_rejects_partial_block() {
        let mut data = q4_0_block(F16_ONE, [0x88; 16]);
        data.pop();
        assert!(dequantize_q4_0(&data).is_err());
    }

    #[test]
    fn q4_k_first_four_scales_and_mins_are_low_6_bits() {
        let mut scales = [0u8; 12];
        scales[1] = 0b11_101010; // top 2 bits belong to sub-block 5
        scales[5] = 0b01_010101;
        assert_eq!(scale_min_k4(1, &scales), (0b101010, 0b010101));
    }

    #[test]
    fn q4_k_last_four_scales_and_mins_are_split_across_bytes() {
        let mut scales = [0u8; 12];
        scales[1] = 0b10_000000; // high 2 bits of scale 5
        scales[5] = 0b01_000000; // high 2 bits of min 5
        scales[9] = 0b0011_1100; // low 4 bits: min 5 (high nibble), scale 5 (low nibble)
        assert_eq!(scale_min_k4(5, &scales), (0b10_1100, 0b01_0011));
    }

    /// The packing side, as written in ggml's quantize_row_q4_K.
    fn pack_scales_mins(sc: [u8; 8], m: [u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for j in 0..8 {
            if j < 4 {
                q[j] = sc[j];
                q[j + 4] = m[j];
            } else {
                q[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
                q[j - 4] |= (sc[j] >> 4) << 6;
                q[j] |= (m[j] >> 4) << 6;
            }
        }
        q
    }

    #[test]
    fn q4_k_scale_min_round_trips_through_ggml_packing() {
        for seed in 0..64u8 {
            let sc: [u8; 8] = std::array::from_fn(|j| (seed + 7 * j as u8) % 64);
            let m: [u8; 8] = std::array::from_fn(|j| (63 - seed + 13 * j as u8) % 64);
            let packed = pack_scales_mins(sc, m);
            for j in 0..8 {
                assert_eq!(
                    scale_min_k4(j, &packed),
                    (sc[j], m[j]),
                    "seed {seed}, j {j}"
                );
            }
        }
    }

    /// One Q4_K super-block.
    fn q4_k_block(d: [u8; 2], dmin: [u8; 2], scales: [u8; 12], qs: [u8; 128]) -> Vec<u8> {
        [d.as_slice(), &dmin, &scales, &qs].concat()
    }

    const F16_ZERO: [u8; 2] = [0, 0];

    #[test]
    fn q4_k_low_nibbles_then_high_nibbles_per_32_bytes() {
        let scales = pack_scales_mins([1; 8], [0; 8]);
        let mut qs = [0u8; 128];
        qs[0] = 0x21; // low -> y[0], high -> y[32]
        qs[32] = 0x43; // low -> y[64], high -> y[96]
        let y = dequantize_q4_k(&q4_k_block(F16_ONE, F16_ZERO, scales, qs)).unwrap();

        assert_eq!((y[0], y[32], y[64], y[96]), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(y.iter().filter(|&&v| v != 0.0).count(), 4);
    }

    #[test]
    fn q4_k_applies_scale_and_min_per_sub_block() {
        let mut sc = [1; 8];
        let mut m = [0; 8];
        (sc[5], m[5]) = (2, 5); // sub-block 5 = y[160..192] = high nibbles of qs[64..96]
        let mut qs = [0u8; 128];
        qs[64] = 0x70;
        let y =
            dequantize_q4_k(&q4_k_block(F16_ONE, F16_ONE, pack_scales_mins(sc, m), qs)).unwrap();

        assert_eq!(y[160], 2.0 * 7.0 - 5.0);
        assert_eq!(y[161], -5.0); // q = 0 still pays the min
        assert_eq!(y[128], 0.0); // sub-block 4, low nibble 0
    }

    #[test]
    fn q4_k_multiple_blocks_are_concatenated() {
        let scales = pack_scales_mins([1; 8], [0; 8]);
        let two = 0x4000u16.to_le_bytes();
        let data = [
            q4_k_block(F16_ONE, F16_ZERO, scales, [0x11; 128]),
            q4_k_block(two, F16_ZERO, scales, [0x11; 128]),
        ]
        .concat();
        let y = dequantize_q4_k(&data).unwrap();

        assert_eq!(y.len(), 512);
        assert_eq!(&y[..256], &[1.0; 256]);
        assert_eq!(&y[256..], &[2.0; 256]);
    }

    #[test]
    fn q4_k_rejects_partial_block() {
        let mut data = q4_k_block(F16_ONE, F16_ZERO, [0; 12], [0; 128]);
        data.pop();
        assert!(dequantize_q4_k(&data).is_err());
    }

    #[test]
    fn f16_normal_numbers() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    }

    #[test]
    fn f16_signed_zero() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn f16_subnormal_numbers() {
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
    }

    #[test]
    fn f16_infinity_and_nan() {
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    /// There are only 65536 f16 values, so check every one against the `half` crate.
    #[test]
    fn f16_matches_half_crate_for_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let expected = half::f16::from_bits(bits).to_f32();
            let actual = f16_to_f32(bits);
            if expected.is_nan() {
                assert!(actual.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{bits:#06x}");
            }
        }
    }
}

/// Cross-check against candle on real files. Each test needs a GGUF that
/// contains tensors of that type:
///
///     GGUF_Q4_0=/path/to/model-Q4_0.gguf \
///     GGUF_Q4_K=/path/to/model-Q4_K_M.gguf \
///         cargo test --release -- --ignored --nocapture
#[cfg(test)]
mod candle_crosscheck {
    use super::*;
    use crate::gguf::{self, GgmlType};
    use candle_core::Device;
    use candle_core::quantized::gguf_file;

    type Dequantize = fn(&[u8]) -> io::Result<Vec<f32>>;

    /// Dequantizes every tensor of type `ty` in $`env` both ways and
    /// requires bit-identical output.
    fn check_every_tensor(
        env: &str,
        ty: GgmlType,
        block: (usize, usize), // (elements, bytes)
        dequantize: Dequantize,
    ) {
        let path = std::env::var(env).unwrap_or_else(|_| panic!("set {env}"));
        let g = gguf::read(&path).unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();

        let mut checked = 0usize;
        let mut elements = 0usize;
        for t in g.tensors.iter().filter(|t| t.ty == ty) {
            let n = t.elements() as usize;
            let len = n / block.0 * block.1;
            let raw = gguf::read_tensor_bytes(&path, &g, t, len).unwrap();
            let ours = dequantize(&raw).unwrap();

            let theirs: Vec<f32> = content
                .tensor(&mut file, &t.name, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            assert_eq!(ours.len(), theirs.len(), "{}", t.name);
            let mismatches = ours
                .iter()
                .zip(&theirs)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(mismatches, 0, "{}: {mismatches}/{n} values differ", t.name);

            checked += 1;
            elements += n;
        }

        let name = ty.name();
        assert!(checked > 0, "no {name} tensors in {path}");
        println!("{checked} {name} tensors, {elements} values: bit-identical to candle");
    }

    #[test]
    #[ignore = "needs a GGUF file with Q4_0 tensors in $GGUF_Q4_0"]
    fn q4_0_matches_candle_for_every_q4_0_tensor() {
        let block = (QK4_0, Q4_0_BLOCK_BYTES);
        check_every_tensor("GGUF_Q4_0", GgmlType(2), block, dequantize_q4_0);
    }

    #[test]
    #[ignore = "needs a GGUF file with Q4_K tensors in $GGUF_Q4_K"]
    fn q4_k_matches_candle_for_every_q4_k_tensor() {
        let block = (QK_K, Q4_K_BLOCK_BYTES);
        check_every_tensor("GGUF_Q4_K", GgmlType(12), block, dequantize_q4_k);
    }
}
