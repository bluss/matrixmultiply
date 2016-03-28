
General matrix multiplication for f32, f64 matrices.

Allows arbitrary row, column strided matrices.

Uses the same microkernel algorithm as BLIS_, but in a much simpler
and less featureful implementation.
See their multithreading_ page for a very good diagram over how
the algorithm partitions the matrix (*Note:* this crate does not implement
multithreading).

.. _BLIS: https://github.com/flame/blis

.. _multithreading: https://github.com/flame/blis/wiki/Multithreading

Please read the `API documentation here`__

__ http://bluss.github.io/matrixmultiply/

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/matrixmultiply.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/matrixmultiply

.. |crates| image:: http://meritbadge.herokuapp.com/matrixmultiply
.. _crates: https://crates.io/crates/matrixmultiply

