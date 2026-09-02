use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug)]
pub struct Content {
    pub version: u32,
    pub metadata_count: u64,
    pub tensor_data_offset: u64,
    pub metadata: HashMap<String, String>,
    pub tensors: Vec<TensorInfo>,
}

pub(crate) struct RawTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: u32,
    pub offset: u64,
}
