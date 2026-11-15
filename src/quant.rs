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
        _ => f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13)),
    }
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
    if data.len() % Q4_0_BLOCK_BYTES != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Q4_0 data must be a multiple of {Q4_0_BLOCK_BYTES} bytes, got {}",
                data.len()
            ),
        ));
    }

    let mut out = Vec::with_capacity(data.len() / Q4_0_BLOCK_BYTES * QK4_0);
    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
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
