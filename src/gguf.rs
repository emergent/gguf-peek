use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

pub const MAGIC: u32 = 0x4655_4747;
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Value::String(s) if s.len() > 60 => format!("{}... ({} bytes)", &s[..60], s.len()),
            Value::String(s) => s.clone(),
            Value::Array(items) => format!("[{} items]", items.len()),
            other => format!("{other:?}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            40 => "NVFP4",
            41 => "Q1_0",
            42 => "Q2_0",
            43 => "F8_E4M3",
            _ => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// `blk.17.attn_v.weight` -> Some((17, "attn_v.weight"))
    pub fn layer(&self) -> Option<(usize, &str)> {
        let rest = self.name.strip_prefix("blk.")?;
        let (num, tail) = rest.split_once('.')?;
        Some((num.parse().ok()?, tail))
    }
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: u64,
}

impl Gguf {
    pub fn alignment(&self) -> u64 {
        self.metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    pub fn block_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.block_count"))?.as_u64()
    }
}

struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    fn bytes(&mut self, n: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf)?;
        self.pos += n as u64;
        Ok(buf)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.bytes(1)?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }

    /// GGUF strings are length-prefixed with a u64 and are NOT NUL-terminated.
    fn string(&mut self) -> io::Result<String> {
        let len = self.u64()? as usize;
        let raw = self.bytes(len)?;
        Ok(String::from_utf8_lossy(&raw).into_owned())
    }

    fn value(&mut self, ty: u32) -> io::Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()?;
                let mut items = Vec::with_capacity(len.min(1024) as usize);
                for _ in 0..len {
                    items.push(self.value(elem_ty)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown metadata value type {other}"),
                ));
            }
        })
    }
}

fn align_up(x: u64, alignment: u64) -> u64 {
    x + (alignment - (x % alignment)) % alignment
}

pub fn read(path: &str) -> io::Result<Gguf> {
    let file = File::open(path)?;
    let mut r = Reader::new(BufReader::new(file));

    let magic = r.u32()?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not a GGUF file (magic = {magic:#010x})"),
        ));
    }

    let version = r.u32()?;
    if version < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF v{version} uses u32 counts; unsupported here"),
        ));
    }

    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        metadata.insert(key, r.value(ty)?);
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ty = GgmlType(r.u32()?);
        let offset = r.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
        });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_ALIGNMENT);

    Ok(Gguf {
        version,
        metadata,
        tensors,
        tensor_data_offset: align_up(r.pos, alignment),
    })
}

/// Reads one tensor's raw (still quantized) bytes.
pub fn read_tensor_bytes(path: &str, g: &Gguf, t: &TensorInfo, len: usize) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(g.tensor_data_offset + t.offset))?;
    let mut buf = vec![0u8; len];
    file.read_exact(&mut buf)?;
    Ok(buf)
}
