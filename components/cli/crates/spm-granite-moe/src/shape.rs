#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoeShape {
    pub width: usize,
    pub ff: usize,
    pub experts: usize,
    pub used: usize,
}

pub const GRANITE: MoeShape = MoeShape {
    width: 1024,
    ff: 512,
    experts: 32,
    used: 8,
};

pub const OLMOE: MoeShape = MoeShape {
    width: 2048,
    ff: 1024,
    experts: 64,
    used: 8,
};

pub fn for_architecture(name: &str) -> Option<MoeShape> {
    match name {
        "granite" | "granitemoe" => Some(GRANITE),
        "olmoe" => Some(OLMOE),
        _ => None,
    }
}

impl MoeShape {
    pub fn validate(self) -> Result<(), String> {
        if self.width == 0 || self.ff == 0 || self.experts == 0 {
            return Err("MoE dimensions must be nonzero".into());
        }
        if self.used == 0 || self.used > self.experts {
            return Err("selected expert count must be within expert count".into());
        }
        if !self.width.is_multiple_of(256) || !self.ff.is_multiple_of(256) {
            return Err("Q6_K matrix dimensions must be multiples of 256".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualification_shapes_are_valid() {
        GRANITE.validate().expect("Granite shape");
        OLMOE.validate().expect("OLMoE shape");
    }

    #[test]
    fn architecture_names_select_shapes_explicitly() {
        assert_eq!(for_architecture("granite"), Some(GRANITE));
        assert_eq!(for_architecture("granitemoe"), Some(GRANITE));
        assert_eq!(for_architecture("olmoe"), Some(OLMOE));
        assert_eq!(for_architecture("dense"), None);
    }
}
