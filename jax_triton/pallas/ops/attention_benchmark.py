import google_benchmark as benchmark

from absl import app
from jax import numpy as jnp
from jax import random
from jax_triton.pallas.ops import attention as tr_attention


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, True])
@benchmark.option.args([128, 2, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
@benchmark.option.args([128, 32, 2048, 512])
@benchmark.option.iterations(100)
def TritonMHA(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  tr_attention.mha(q, k, v, causal=False, block_q=128, block_k=64)

  while benchmark_state:
      o_ref = tr_attention.mha(q, k, v, causal=False, block_q=128, block_k=64)


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, True])
@benchmark.option.args([128, 2, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
@benchmark.option.args([128, 32, 2048, 512])
@benchmark.option.iterations(100)
def BaseLiMHA(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  tr_attention.mha_reference(q, k, v, causal=False)

  while benchmark_state:
    o_ref = tr_attention.mha_reference(q, k, v, causal=False)


if __name__ == "__main__":
  app.run(benchmark.main)
