## Why

Currently, evaluating OpenJevPro requires a running vLLM, SGLang, Ollama, or Laya inference endpoint with logprobs and lexical grammar masking enabled. For developers browsing GitHub or evaluating agent routing libraries on laptops without active GPU servers, this dependency creates an impassable "Time-to-Hello-World" friction point, causing majority evaluation drop-off before first execution.

Introducing a zero-GPU, zero-dependency mock client and built-in CLI playground enables any developer to install and verify deterministic decision primitives, calibrated confidence, and order-invariance in under 5 seconds on pure Python/CPU.

## What Changes

- Introduce `openjevpro.mock.MockClient` and `OpenJevProClient(mock=True)` providing drop-in offline simulation of discrete choice routing, binary assertion (`Noul`), and graded scoring (`Score`).
- Implement simulated commutative softmax and isolated candidate scoring in the mock runtime to demonstrate mathematical permutation invariance without backend connectivity.
- Provide a zero-argument CLI demonstration command `python -m openjevpro.demo` that executes sample decisions and prints calibrated receipts directly in the terminal.
- Support pre-canned query rules and deterministic fuzzy logit synthesis for fast agent unit testing in CI/CD without spinning up mock HTTP servers.

## Capabilities

### New Capabilities
- `zero-gpu-mock-runtime`: In-memory deterministic mock decision engine, `MockClient` interface, and zero-argument CLI demo runner providing immediate local developer evaluation.

### Modified Capabilities
<!-- None. Existing spec requirements remain intact. -->

## Impact

- Affected Code:
  - New `openjevpro/mock.py` containing `MockClient` and deterministic logit synthesizer.
  - Updated `openjevpro/__init__.py` exporting `MockClient`.
  - Updated `openjevpro/client.py` accepting optional `mock=True` parameter.
  - Updated `openjevpro/demo.py` defaulting to `MockClient` if no live server endpoint is provided.
- Dependencies: Zero new external dependencies (pure Python standard library).
- Compatibility: 100% backward compatible with existing client and harness APIs.
