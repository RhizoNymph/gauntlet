//! Pure analysis logic: fleet statistics, network model fitting, and the
//! pairwise tournament scheduler. Everything here is deterministic and
//! side-effect free.

pub mod fit;
pub mod schedule;
pub mod skew;
pub mod stats;
