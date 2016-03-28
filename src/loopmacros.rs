
macro_rules! loop4 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
    }}
}

macro_rules! loop4x4 {
    ($i:ident, $j:ident, $e:expr) => {{
        loop4!($i, loop4!($j, $e));
    }}
}

macro_rules! unroll_by_8 {
    ($ntimes:expr, $e:expr) => {{
        let k = $ntimes;
        for _ in 0..k / 8 {
            $e;$e; $e;$e;
            $e;$e; $e;$e;
        }
        for _ in 0..k % 8 {
            $e
        }
    }}
}

macro_rules! shuf {
    ($v:expr, $i:expr, $j:expr, $k:expr, $m:expr) => (
        [$v[$i], $v[$j], $v[$k], $v[$m]]
    );
}

