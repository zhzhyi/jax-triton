'''
Results on 1 A100

--------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations
--------------------------------------------------------------------------------
TritFWD/16/4/64/1/iterations:100           0.033 ms        0.033 ms          100
TritFWD/64/2/2048/2/iterations:100         0.191 ms        0.044 ms          100
TritFWD/64/12/2048/2/iterations:100        0.354 ms        0.045 ms          100
TritFWD/128/8/2048/2/iterations:100        0.492 ms        0.052 ms          100
TritFWD/128/16/2048/2/iterations:100       0.853 ms        0.065 ms          100
TritFWD/128/32/2048/2/iterations:100        1.68 ms        0.204 ms          100
TritFWD/128/32/2048/32/iterations:100       19.3 ms        0.196 ms          100
TritFWD/128/32/2048/64/iterations:100       38.2 ms        0.259 ms          100
BaseFWD/16/4/64/1/iterations:100           0.074 ms        0.072 ms          100
BaseFWD/64/2/2048/2/iterations:100         0.460 ms        0.116 ms          100
BaseFWD/64/12/2048/2/iterations:100         1.40 ms        0.250 ms          100
BaseFWD/128/2/2048/2/iterations:100        0.410 ms        0.145 ms          100
BaseFWD/128/8/2048/2/iterations:100         1.07 ms        0.154 ms          100
BaseFWD/128/16/2048/2/iterations:100        1.98 ms        0.280 ms          100
BaseFWD/128/32/2048/2/iterations:100        3.51 ms        0.355 ms          100
BaseFWD/128/32/2048/32/iterations:100       44.9 ms        0.326 ms          100
TritBWD/16/4/64/1/iterations:100           0.066 ms        0.066 ms          100
TritBWD/64/2/2048/2/iterations:100          2.58 ms        0.193 ms          100
TritBWD/64/12/2048/2/iterations:100         3.29 ms        0.191 ms          100
BaseBWD/16/4/64/1/iterations:100           0.101 ms        0.101 ms          100
BaseBWD/64/2/2048/2/iterations:100         0.335 ms        0.163 ms          100
BaseBWD/64/12/2048/2/iterations:100         1.52 ms        0.252 ms          100
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

  f = jax.jit(partial(attention.mha, num_stages=1))
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

  f_grad = jax.jit(jax.grad(f, argnums=(0, 1, 2)))

  dq, dk, dv = f_grad(q, k, v)

  while benchmark_state:
    dq, dk, dv = f_grad(q, k, v)


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

  f_grad = jax.jit(jax.grad(f_ref, argnums=(0, 1, 2)))

  dq_ref, dk_ref, dv_ref = f_grad(q, k, v)

  while benchmark_state:
    dq_ref, dk_ref, dv_ref = f_grad(q, k, v)


if __name__ == "__main__":
  app.run(benchmark.main)
