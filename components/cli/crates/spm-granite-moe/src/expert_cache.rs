use spm_gguf::Content;
use std::{collections::BTreeMap, path::Path, sync::Arc};

pub struct ExpertCache {
    budget: usize,
    resident: usize,
    age: u64,
    seen: BTreeMap<String, u64>,
    entries: BTreeMap<String, Entry>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

struct Entry {
    bytes: Arc<Vec<u8>>,
    age: u64,
}

impl ExpertCache {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            resident: 0,
            age: 0,
            seen: BTreeMap::new(),
            entries: BTreeMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    pub fn get(
        &mut self,
        model: &Path,
        content: &Content,
        name: &str,
        expert: usize,
        size: usize,
    ) -> Result<Arc<Vec<u8>>, String> {
        let key = format!("{name}:{expert}");
        self.age += 1;
        if let Some(entry) = self.entries.get_mut(&key) {
            self.hits += 1;
            entry.age = self.age;
            return Ok(Arc::clone(&entry.bytes));
        }
        self.misses += 1;
        let bytes = Arc::new(crate::gemma_smoke::expert_bytes(
            model, content, name, expert, size,
        )?);
        let seen = self.seen.entry(key.clone()).or_default();
        *seen += 1;
        if *seen >= 2 && size <= self.budget {
            while self.resident + size > self.budget {
                self.evict()?;
            }
            self.resident += size;
            self.entries.insert(
                key,
                Entry {
                    bytes: Arc::clone(&bytes),
                    age: self.age,
                },
            );
        }
        Ok(bytes)
    }

    fn evict(&mut self) -> Result<(), String> {
        let key = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.age)
            .map(|(key, _)| key.clone())
            .ok_or("cache cannot satisfy budget")?;
        let entry = self
            .entries
            .remove(&key)
            .ok_or("cache victim disappeared")?;
        self.resident -= entry.bytes.len();
        self.evictions += 1;
        Ok(())
    }

    pub fn resident(&self) -> usize {
        self.resident
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn starts_empty_and_bounded() {
        let cache = ExpertCache::new(128 * 1024 * 1024);
        assert_eq!((cache.resident(), cache.hits, cache.misses), (0, 0, 0));
    }
}
