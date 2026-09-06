const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

pub struct Digest(u64);

impl Digest {
    pub const fn new() -> Self {
        Self(OFFSET)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(PRIME);
        }
    }

    pub const fn value(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::Digest;

    #[test]
    fn fnv_known_answer() {
        let mut digest = Digest::new();
        digest.update(b"hello");
        assert_eq!(digest.value(), 0xa430_d846_80aa_bd0b);
    }
}
