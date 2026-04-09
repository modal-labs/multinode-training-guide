from math import sqrt
import jax
import equinox as eqx
from jax.lax import sub
from matplotlib.pyplot import flag


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key, 2)
        self.weight = jax.random.normal(wkey, (in_size, out_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return x @ self.weight + self.bias


class GroupNorm(eqx.Module):
    gamma: jax.Array
    beta: jax.Array
    num_groups: int
    epsilon: float

    def __init__(self, in_shape, out_shape, num_groups, key):
        gkey, bkey = jax.random.split(key, 2)
        self.gamma = jax.random.normal(gkey, in_shape)
        self.beta = jax.random.normal(bkey, out_shape)
        self.num_groups = num_groups
        self.epsilon = 1e-5

    # shape is defined as (N, C, X, Y) from the original GroupNorm paper
    def __call__(self, x: jax.Array):
        x = x.reshape(
            (
                x.shape[0],
                self.num_groups,
                x.shape[1] // self.num_groups,
                x.shape[2],
                x.shape[3],
            )
        )
        mean = jax.numpy.mean(x, axis=(2, 3, 4), keepdims=True)
        var = jax.numpy.var(x, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / jax.numpy.sqrt(var + self.epsilon)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        return x * self.gamma + self.beta


class BatchNorm(eqx.Module):
    gamma: jax.Array
    beta: jax.Array
    epsilon: float

    def __init__(self, in_shape, out_shape, key):
        gkey, bkey = jax.random.split(key, 2)
        self.gamma = jax.random.normal(gkey, in_shape)
        self.beta = jax.random.normal(bkey, out_shape)
        self.epsilon = 1e-5

    # TODO(atoniolo76): add flag for inference time vs. training time
    def __call__(
        self,
        x,
    ):
        mean = jax.numpy.mean(x, axis=0)
        var = jax.numpy.var(x, axis=0)
        return (
            (x - mean) / jax.numpy.sqrt(var + self.epsilon)
        ) * self.gamma + self.beta


class MLP(eqx.Module):
    layers: dict
    hidden_dim: int

    def __init__(self, in_shape, hidden_dim, out_shape, key):
        lkey, hkey, okey = jax.random.split(key, 3)
        self.hidden_dim = hidden_dim
        layers = dict()
        layers["input"] = Linear(in_shape, hidden_dim, lkey)
        layers["batchnorm"] = BatchNorm(hidden_dim, hidden_dim, hkey)
        layers["output"] = Linear(hidden_dim, out_shape, okey)
        self.layers = layers

    def __call__(self, x):
        return self.layers["output"](
            jax.numpy.tanh(self.layers["batchnorm"](self.layers["input"](x)))
        )
