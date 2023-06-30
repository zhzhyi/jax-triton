'''
Results on 1 A100

--------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations
--------------------------------------------------------------------------------
TritFWD/16/4/64/1/iterations:100           0.373 ms        0.373 ms          100
TritFWD/64/2/2048/2/iterations:100         0.477 ms        0.415 ms          100
TritFWD/64/12/2048/2/iterations:100        0.710 ms        0.453 ms          100
TritFWD/128/8/2048/2/iterations:100        0.831 ms        0.409 ms          100
TritFWD/128/16/2048/2/iterations:100        1.46 ms        0.750 ms          100
TritFWD/128/32/2048/2/iterations:100        2.09 ms        0.583 ms          100
TritFWD/128/32/2048/32/iterations:100       19.8 ms        0.733 ms          100
TritFWD/128/32/2048/64/iterations:100       38.6 ms        0.720 ms          100
BaseFWD/16/4/64/1/iterations:100           0.071 ms        0.069 ms          100
BaseFWD/64/2/2048/2/iterations:100         0.550 ms        0.087 ms          100
BaseFWD/64/12/2048/2/iterations:100         1.60 ms        0.148 ms          100
BaseFWD/128/2/2048/2/iterations:100        0.437 ms        0.138 ms          100
BaseFWD/128/8/2048/2/iterations:100         1.05 ms        0.106 ms          100
BaseFWD/128/16/2048/2/iterations:100        1.95 ms        0.287 ms          100
BaseFWD/128/32/2048/2/iterations:100        3.35 ms        0.263 ms          100
BaseFWD/128/32/2048/32/iterations:100       44.9 ms        0.338 ms          100
TritBWD/16/4/64/1/iterations:100            1.78 ms         1.78 ms          100
TritBWD/64/2/2048/2/iterations:100          3.20 ms         1.95 ms          100
TritBWD/64/12/2048/2/iterations:100         4.05 ms         1.97 ms          100
BaseBWD/16/4/64/1/iterations:100            1.97 ms         1.97 ms          100
BaseBWD/64/2/2048/2/iterations:100          1.93 ms         1.93 ms          100
BaseBWD/64/12/2048/2/iterations:100         2.81 ms         2.12 ms          100
'''
import google_benchmark as benchmark
import jax

from absl import app
from functools import partial
from jax import numpy as jnp
from jax import random
from jax_triton.pallas.ops import attention


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, 1])
@benchmark.option.args([64, 2, 2048, 2])
@benchmark.option.args([64, 12, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
@benchmark.option.args([128, 32, 2048, 64])
@benchmark.option.iterations(100)
def TritFWD(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  f = partial(attention.mha, num_stages=1)
  o = f(q, k, v).block_until_ready()

  while benchmark_state:
    o = f(q, k, v).block_until_ready()


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, 1])
@benchmark.option.args([64, 2, 2048, 2])
@benchmark.option.args([64, 12, 2048, 2])
@benchmark.option.args([128, 2, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
# @benchmark.option.args([128, 32, 2048, 64]) OOM
@benchmark.option.iterations(100)
def BaseFWD(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  f = jax.jit(attention.mha_reference)

  o = f(q, k, v).block_until_ready()

  while benchmark_state:
    o = f(q, k, v).block_until_ready()


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, 1])
@benchmark.option.args([64, 2, 2048, 2])
@benchmark.option.args([64, 12, 2048, 2])
@benchmark.option.iterations(100)
def TritBWD(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  @jax.jit
  def f(q, k, v):
    return attention.mha(q, k, v).sum()

  dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)

  while benchmark_state:
    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, 1])
@benchmark.option.args([64, 2, 2048, 2])
@benchmark.option.args([64, 12, 2048, 2])
@benchmark.option.iterations(100)
def BaseBWD(benchmark_state: benchmark.State):
  k1, k2, k3 = random.split(random.PRNGKey(0), 3)


  head_dim = benchmark_state.range(0)
  num_heads = benchmark_state.range(1)
  seq_len = benchmark_state.range(2)

  batch_size = benchmark_state.range(3)
  q = random.normal(k1, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  k = random.normal(k2, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)
  v = random.normal(k3, (batch_size, seq_len, num_heads, head_dim), dtype=jnp.float16)

  @jax.jit
  def f_ref(q, k, v):
    return attention.mha_reference(q, k, v).sum()

  dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)

  while benchmark_state:
    dq_ref, dk_ref, dv_ref = jax.grad(f_ref, argnums=(0, 1, 2))(q, k, v)


if __name__ == "__main__":
  app.run(benchmark.main)
