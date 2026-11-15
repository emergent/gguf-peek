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

#[cfg(test)]
mod tests {
    use super::*;

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
