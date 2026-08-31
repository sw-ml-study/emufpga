//! HRM's shape, from `zbloss/HRM-sudoku-extreme`'s `config.json`.

use spm_trm::TrmConfig;

/// The recursion and module shape.
///
/// Block shape is [`TrmConfig`], which HRM and TRM share -- same
/// hidden size, heads, expansion, intermediate width, `RoPE` base and
/// norm epsilon. Only the recursion differs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HrmConfig {
    /// Shape of one transformer block.
    pub block: TrmConfig,
    /// Layers in the low-level module.
    pub low_layers: usize,
    /// Layers in the high-level module.
    pub high_layers: usize,
    /// Outer cycles.
    pub h_cycles: usize,
    /// Low-level updates per outer cycle.
    pub l_cycles: usize,
}

impl Default for HrmConfig {
    /// `zbloss/HRM-sudoku-extreme`'s published values.
    fn default() -> Self {
        Self {
            block: TrmConfig {
                layers: 4,
                ..TrmConfig::default()
            },
            low_layers: 4,
            high_layers: 4,
            h_cycles: 2,
            l_cycles: 2,
        }
    }
}

impl HrmConfig {
    /// Streams in the rotating region: four projections per layer,
    /// across both modules.
    #[must_use]
    pub const fn rotating_streams(&self) -> usize {
        (self.low_layers + self.high_layers) * 4
    }

    /// Module sweeps per forward: `h_cycles` low sweeps times
    /// `l_cycles`, plus one high sweep per outer cycle.
    #[must_use]
    pub const fn sweeps(&self) -> usize {
        self.h_cycles * (self.l_cycles + 1)
    }

    /// Rewinds a forward pass issues.
    ///
    /// One before every low sweep except the very first. The high
    /// sweep never needs one: the last low sweep of an outer cycle
    /// leaves the cursor exactly where the high module begins.
    #[must_use]
    pub const fn rewinds(&self) -> usize {
        self.h_cycles * self.l_cycles - 1
    }
}
