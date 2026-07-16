// Copyright 2025 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// This document retraces the vector permutations in the x86-64 AVX sgemm microkernel,
// to verify and visualize where the elements from the input buffers end up.

#set document(
  date: none,
  author: ("Ulrik Sverdrup", ),
  title: "matrixmultiply: x86-64 AVX sgemm microkernel",
)

#set text(font: "Fira Sans", size: 11pt, features: ())
#let rawfont = "Fira Code"
#show raw: set text(font: rawfont, size: 10pt)

#show link: underline.with(evade: false)
#set page(numbering: "1", header: {
  set align(right)
  set text(size: 0.8em)
  [matrixmultiply #link("https://github.com/bluss/matrixmultiply")]
})


/// Add string prefix to each array element
#let tag(name, arr) = {
  arr.map(x => name + str(x))
}

#let load_ps(name) = {
  tag(name, range(0, 8))
}

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_moveldup_ps&ig_expand=4923,6050,4597
#let moveldup_ps(x) = {
  range(0, x.len()).map(i => x.at(2 * calc.div-euclid(i, 2)))
}

#let movehdup_ps(x) = {
  range(0, x.len()).map(i => x.at(1 + 2 * calc.div-euclid(i, 2)))
}

#let select4_128(src, control) = {
  let i = control
  if i <= 3 {
    src.slice(i, i + 1)
  } else {
    panic("invalid control")
  }
}


/// _mm256_permute_ps
/// control word a, b, c, d (each 2 bits)
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_permute_ps&ig_expand=4923
#let permute_ps(x, a, b, c, d) = {
  for (i, c) in (a, b, c, d).enumerate() {
    select4_128(x.slice(0, 4), c)
  }
  for (i, c) in (a, b, c, d).enumerate() {
    select4_128(x.slice(4, 8), c)
  }
}

/// _mm256_permute2f128_ps
/// control word a, b (each 2 bits)
#let permute2f128_ps(src1, src2, a, b) = {
  let select4_perm(control) = {
    if control == 0 {
      src1.slice(0, 4)
    } else if control == 1 {
      src1.slice(4, 8)
    } else if control == 2 {
      src2.slice(0, 4)
    } else if control == 3 {
      src2.slice(4, 8)
    } else {
      panic("invalid control")
    }
  }
  select4_perm(a)
  select4_perm(b)
}

/// _mm256_shuffle_ps
/// control word a, b, c, d (each 2 bits)
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_shuffle_ps&ig_expand=4923,6050
#let shuffle_ps(src1, src2, a, b, c, d) = {
  let control-and-source = (a, b, c, d).zip((src1, src1, src2, src2)).enumerate()
  for (i, (c, src)) in control-and-source {
    select4_128(src.slice(0, 4), c)
  }
  for (i, (c, src)) in control-and-source {
    select4_128(src.slice(4, 8), c)
  }
}



#let digits = "0123456789".codepoints()
/// Translate a1b2 to ab12
#let norm-name(x) = {
  x.split("").sorted(key: x => digits.contains(x)).join()
}

/// Multiply two arrays  (a0, a1) * (b0, b1) == (a0b0, a1b1)
#let mul(x, y) = {
  x.zip(y, exact: true).map(((a, b)) => a + b).map(norm-name)
}

/// Map array (of string) to (elt, bool) where the boolean marks it as duplicated or not
#let markduplicates(arr) = {
  let counter = (:)
  for elt in arr {
    let c = 1 + counter.at(elt, default: 0)
    counter.insert(elt, c)
  }
  arr.map(elt => (elt, counter.at(elt) > 1))
}


#let show-vectors(ab, name: none, row-label: none, check-duplicates: true) = {
  let ncol = 8
  let vector-width = 3.5em
  let color-indices = true

  let elements = ab.flatten()
  let extra-col = 0
  let nrows = calc.div-euclid(ab.flatten().len(), 8)

  let row-enumerator = box
  if name != none and row-label == none {
    row-label = name
    row-enumerator = x => none
  } else if name != none {
    block(strong(name), below: 0.6em)
  }

  show sub: text.with(size: 1.3em)
  show <row-label>: it => {
    set text(font: rawfont, size: 9pt)
    strong(it.body)
  }

  show table.cell: it => {
    if it.x >= ncol {
      return it
    }
    show regex("([a-z]+[0-9]*)+"): it => {
      show regex("\d"): it => {
        let color = if not color-indices {
          black
        } else if it.text.match(regex("[37]")) != none {
          green.darken(10%)
        } else if it.text.match(regex("[15]")) != none {
          red.darken(20%)
        } else if it.text.match(regex("[26]")) != none {
          blue.darken(10%)
        } else {
          black
        }
        set text(fill: color)
        strong(sub(it))
      }
      it
    }
    it
  }


  // check and mark duplicates
  if nrows > 1 and check-duplicates {
    elements = markduplicates(elements).map(((elt, duplicated)) => {
      set text(stroke: red + 0.7pt) if duplicated
      elt
    })
  }

  if row-label != none {
    elements = elements.chunks(8).enumerate().map(
      ((i, c)) => c + ([_#row-label;#row-enumerator[[#i]]_<row-label>], )
    ).flatten()
    extra-col += 1
  }
  let t = 0.5pt
  table(
    columns: (vector-width,) * ncol + (auto, ) * extra-col,
    align: bottom + center,
    inset: (bottom: 0.5em),
    stroke: (x, y) => {
      let st = (:)
      if x == 0 { st.insert("left", t) }
      if x == ncol - 1 { st.insert("right", t) }
      if y == 0 and x < ncol { st.insert("top", t)}
      if y == nrows - 1 and x < ncol { st.insert("bottom", t) }
      st
    },
    fill: (x, y) => if x >= 8 { none } else if calc.odd(y) { rgb("EAF2F5") },
    ..elements,
    table.vline(x: 2, position: start, stroke: t / 4),
    table.vline(x: 4, position: start, stroke: t / 2),
    table.vline(x: 6, position: start, stroke: t / 4),
  )
}


= x86-64 AVX/FMA sgemm microkernel: 32-bit float

== Loop Iteration

Load data from buffers `a` and `b` into vectors `aNNNN` and `bv`, `bv_lh`.
#{
  let av = load_ps("a")
  let bv = load_ps("b")
  let a0246 = moveldup_ps(av)
  let a2064 = permute_ps(a0246, 2, 3, 0, 1)
  let a1357 = movehdup_ps(av)
  let a3175 = permute_ps(a1357, 2, 3, 0, 1)
  let bv_lh = permute2f128_ps(bv, bv, 3, 0)

  show-vectors(av, name: `av`)
  show-vectors(a0246, name: `a0246`)
  show-vectors(a2064, name: `a2064`)
  show-vectors(a1357, name: `a1357`)
  show-vectors(a3175, name: `a3175`)
  show-vectors(bv, name: `bv`)
  show-vectors(bv_lh, name: `bv_lh`)

  [
    #show "+=": $+#h(0em)=$
    #show "*": $times$
    ```rust
    ab[0] += a0246 * bv
    ab[1] += a2064 * bv
    ab[2] += a0246 * bv_lh
    ab[3] += a2064 * bv_lh
    ab[4] += a1357 * bv
    ab[5] += a3175 * bv
    ab[6] += a1357 * bv_lh
    ab[7] += a3175 * bv_lh
    ```
  ]

  let ab = (
    mul(a0246, bv),
    mul(a2064, bv),
    mul(a0246, bv_lh),
    mul(a2064, bv_lh),

    mul(a1357, bv),
    mul(a3175, bv),
    mul(a1357, bv_lh),
    mul(a3175, bv_lh),
  )

  show-vectors(ab, name: [`ab` accumulator in loop], row-label: [ab])
  if ab.flatten().len() != ab.flatten().dedup().len() {
    highlight(fill: red, [Duplicate entries])
  }

  pagebreak()

  [
    == Finish
    De-stripe data from accumulator into final storage order.
  ]

  let shuf_mask = (0, 1, 2, 3)
  let shuffle_ab = (i, j) => shuffle_ps(ab.at(i), ab.at(j), ..shuf_mask)
  let ab0044 = shuffle_ab(0, 1)
  let ab2266 = shuffle_ab(1, 0)
  let ab4400 = shuffle_ab(2, 3)
  let ab6622 = shuffle_ab(3, 2)

  let ab1155 = shuffle_ab(4, 5)
  let ab3377 = shuffle_ab(5, 4)
  let ab5511 = shuffle_ab(6, 7)
  let ab7733 = shuffle_ab(7, 6)

  show-vectors(ab0044, name: `ab0044`)
  show-vectors(ab2266, name: `ab2266`)
  show-vectors(ab4400, name: `ab4400`)
  show-vectors(ab6622, name: `ab6622`)

  show-vectors(ab1155, name: `ab1155`)
  show-vectors(ab3377, name: `ab3377`)
  show-vectors(ab5511, name: `ab5511`)
  show-vectors(ab7733, name: `ab7733`)

  let abfinal = (
    permute2f128_ps(ab0044, ab4400, 0, 2),
    permute2f128_ps(ab1155, ab5511, 0, 2),
    permute2f128_ps(ab2266, ab6622, 0, 2),
    permute2f128_ps(ab3377, ab7733, 0, 2),
    permute2f128_ps(ab0044, ab4400, 3, 1),
    permute2f128_ps(ab1155, ab5511, 3, 1),
    permute2f128_ps(ab2266, ab6622, 3, 1),
    permute2f128_ps(ab3377, ab7733, 3, 1),
  )

  show-vectors(abfinal, name: [`ab` in order], row-label: [ab])
  if abfinal.flatten().len() != abfinal.flatten().dedup().len() {
    highlight(fill: red, [Duplicate entries])
  }
}
