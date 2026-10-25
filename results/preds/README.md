Stored predictions of every training run, one `.npz` file per run (named after the run identifier, with `|`
replaced by `__`). Each file holds the test and validation probabilities (`P_test`, `P_test_plain`, `P_val`),
the true labels, the image keys (`category/file`), and the leaf group and session of each test image.
