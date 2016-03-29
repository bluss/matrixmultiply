pub const S_ALIGN: usize = 32;
pub const S_MR: usize = 8;
pub const S_NR: usize = 8;
pub const D_ALIGN: usize = 32;
pub const D_MR: usize = 8;
pub const D_NR: usize = 4;

pub use bli_sgemm_asm_8x8 as sgemm_asm;
pub use bli_dgemm_asm_8x4 as dgemm_asm;

#[allow(non_camel_case_types)]
pub type dim_t = isize;
#[allow(non_camel_case_types)]
pub type inc_t = isize;

extern {
    pub fn bli_sgemm_asm_8x8(
                          k: dim_t,
                          alpha: *const f32,
                          a: *const f32,
                          b: *const f32,
                          beta: *const f32,
                          c: *mut f32,
                          rs_c: inc_t, cs_c: inc_t
                         );
    pub fn bli_dgemm_asm_8x4(
                          k: dim_t,
                          alpha: *const f64,
                          a: *const f64,
                          b: *const f64,
                          beta: *const f64,
                          c: *mut f64,
                          rs_c: inc_t, cs_c: inc_t
                         );
}
