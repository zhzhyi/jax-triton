import google_benchmark as benchmark
import jax

from absl import app
from functools import partial
from jax import numpy as jnp
from jax import random
from jax_triton.pallas.ops import attention


@benchmark.register
@benchmark.option.unit(benchmark.kMillisecond)
@benchmark.option.args([16, 4, 64, True])
@benchmark.option.args([128, 2, 2048, 2])
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
@benchmark.option.args([16, 4, 64, True])
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
@benchmark.option.args([16, 4, 64, True])
@benchmark.option.args([128, 2, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
@benchmark.option.args([128, 32, 2048, 64])
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
@benchmark.option.args([16, 4, 64, True])
@benchmark.option.args([128, 2, 2048, 2])
@benchmark.option.args([128, 8, 2048, 2])
@benchmark.option.args([128, 16, 2048, 2])
@benchmark.option.args([128, 32, 2048, 2])
@benchmark.option.args([128, 32, 2048, 32])
# @benchmark.option.args([128, 32, 2048, 64]) OOM
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
