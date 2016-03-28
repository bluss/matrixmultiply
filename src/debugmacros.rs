
// for debugging -- like println during debugging
#[cfg(debug_assertions)]
macro_rules! dprint {
    ($($t:tt)*) => {
        println!($($t)*)
    }
}

#[cfg(not(debug_assertions))]
macro_rules! dprint {
    ($($t:tt)*) => {
    }
}

#[cfg(debug_assertions)]
macro_rules! debug {
    ($e:expr) => {
        $e;
    }
}

#[cfg(not(debug_assertions))]
macro_rules! debug {
    ($e:expr) => {
    }
}

