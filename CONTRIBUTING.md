# Contributing to specbench

Contributions should make correctness, measurement, or reproduction easier to assess.

1. Start with the README's CPU or GPU path in an isolated environment. Report the Python/PyTorch/CUDA/driver versions and hardware; do not treat a new setup as an exact reproduction of the published laptop environment.
2. For a correctness change, describe a minimal failing case and add a focused test. Run the relevant synthetic tests before benchmarking. GPU/cache/distribution changes also need the applicable real-model tests.
3. For a performance claim, keep the target model, prompt set, decoding settings, and token budget matched between baseline and experimental arms. Preserve raw paired records and uncertainty intervals.
4. Include the problem, implementation, validation, and known limits in a focused pull request.

Good first work includes setup documentation, additional sampler distributions, cache edge cases, and clearly labeled hardware reproductions. Do not overwrite published raw records to make a new experiment appear to be the old one. Keep model weights, credentials, private prompts, and large hidden-state caches out of commits.

Synthetic checks are empirical implementation tests; distinguish them from theoretical claims and real-model distribution validation. Negative speedups and failed configurations are useful results.

No standalone LICENSE file is currently included. Discuss reuse terms and identify third-party sources before introducing external code or datasets.
