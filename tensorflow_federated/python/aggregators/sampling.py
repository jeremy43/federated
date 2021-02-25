# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregator for sampling client values."""

import collections
from typing import Optional

import tensorflow as tf

from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.common_libs import py_typecheck
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.core.api import computation_base
from tensorflow_federated.python.core.api import computation_types
from tensorflow_federated.python.core.api import computations
from tensorflow_federated.python.core.api import intrinsics
from tensorflow_federated.python.core.api import placements
from tensorflow_federated.python.core.impl.types import type_transformations
from tensorflow_federated.python.core.templates import aggregation_process
from tensorflow_federated.python.core.templates import measured_process


def _build_reservoir_type(
    sample_value_type: computation_types.Type) -> computation_types.Type:
  """Create the TFF type for the reservoir structure."""

  def add_unknown_dimension(t):
    if not (t.is_tensor() or t.is_struct()):
      raise TypeError('Cannot create a reservoir on type structure, must only '
                      'contain `TensorType` or `StructType`, had a '
                      f'{type(t)!r}.')
    if t.is_tensor():
      return (computation_types.TensorType(
          dtype=t.dtype, shape=[None] + t.shape), True)
    return t, False

  return computation_types.to_type(
      collections.OrderedDict(
          random_seed=(computation_types.TensorType(tf.int64),
                       computation_types.TensorType(tf.int64)),
          random_values=computation_types.TensorType(tf.int32, shape=[None]),
          samples=type_transformations.transform_type_postorder(
              sample_value_type, add_unknown_dimension)[0]))


def _build_initial_sample_reservoir(sample_value_type: computation_types.Type,
                                    seed: Optional[int] = None):  # pylint: disable=unused-argument
  """Build up the initial state of the reservoir for sampling.

  Args:
    sample_value_type: The type structure of the values that will be sampled.
    seed: An optional tensor, or Python value convertible to a tensor, that
      serves as the initial seed to the random process.

  Returns:
    A structure containing the samples and metadata for reservoir sampling
    during the three stages of `tff.federated_aggregate`.
  """

  @computations.tf_computation
  def initialize():
    # Allow fixed seeds, otherwise use the current timestamp.
    nonlocal seed
    if seed is None:
      seed = tf.cast(tf.timestamp() * 1000.0, tf.int64)
    elif tf.is_tensor(seed):
      if seed.dtype != tf.int64:
        seed = tf.cast(seed, dtype=tf.int64)
    else:
      seed = tf.convert_to_tensor(seed, dtype=tf.int64)
    # Build a "container" tensor, possibly inside a structure.
    def zero_for_tensor_type(t: computation_types.TensorType):
      if not t.is_tensor():
        raise TypeError(f'Cannot create zero for non TesnorType: {type(t)}')
      return tf.zeros([0] + t.shape, dtype=t.dtype)

    if sample_value_type.is_tensor():
      initial_samples = zero_for_tensor_type(sample_value_type)
    elif sample_value_type.is_struct():
      initial_samples = structure.map_structure(zero_for_tensor_type,
                                                sample_value_type)
    else:
      raise TypeError('Cannot build initial reservoir for structure that has '
                      'not solely StructType or TensorType, got '
                      f'{sample_value_type!r}.')
    return collections.OrderedDict(
        random_seed=(seed, seed),
        random_values=tf.zeros([0], tf.int32),
        samples=initial_samples)

  return initialize()


def _build_sample_value(value_type: computation_types.Type,
                        sample_size: int) -> computation_base.Computation:
  """Builds the `accumulate` computation for sampling."""
  reservoir_type = _build_reservoir_type(value_type)

  def add_sample(reservoir, new_seed, sample_random_value, sample):
    new_random_values = tf.concat(
        [reservoir['random_values'], sample_random_value], axis=0)
    new_samples = tf.nest.map_structure(
        lambda a, b: tf.concat([a, tf.expand_dims(b, axis=0)], axis=0),
        reservoir['samples'], sample)
    return collections.OrderedDict(
        random_seed=new_seed,
        random_values=new_random_values,
        samples=new_samples)

  def pop_one_minimum_value(reservoir):
    """Remove one element from the reservoir based on the minimum value."""
    size_after_pop = tf.size(reservoir['random_values']) - 1
    _, indices = tf.nn.top_k(
        reservoir['random_values'], k=size_after_pop, sorted=False)
    return collections.OrderedDict(
        random_seed=reservoir['random_seed'],
        random_values=tf.gather(reservoir['random_values'], indices),
        samples=tf.nest.map_structure(lambda t: tf.gather(t, indices),
                                      reservoir['samples']))

  @computations.tf_computation(reservoir_type, value_type)
  @tf.function
  def perform_sampling(reservoir, sample):
    # Pick a new random number for the incoming sample, and advance the seed
    # for the next sample.
    seed = reservoir['random_seed']
    sample_random_value = tf.random.stateless_uniform(
        shape=(1,), minval=None, seed=seed, dtype=tf.int32)
    new_seed = (seed[0], tf.squeeze(tf.cast(sample_random_value, tf.int64)))
    # If the reservoir isn't full, add the sample.
    if tf.less(tf.size(reservoir['random_values']), sample_size):
      return add_sample(reservoir, new_seed, sample_random_value, sample)
    else:
      # Determine if the random value for this sample belongs in the reservoir:
      # random value larger than the smallest see so far. Or if the sample
      # should be discarded: its random value is smaller than the smallest we've
      # already seen.
      min_reservoir_value = tf.reduce_min(reservoir['random_values'])
      if sample_random_value < min_reservoir_value:
        return collections.OrderedDict(reservoir, random_seed=new_seed)
      reservoir = pop_one_minimum_value(reservoir)
      return add_sample(reservoir, new_seed, sample_random_value, sample)

  return perform_sampling


def _build_merge_samples(value_type: computation_types.Type,
                         sample_size: int) -> computation_base.Computation:
  """Builds the `merge` computation for a sampling."""
  reservoir_type = _build_reservoir_type(value_type)

  @computations.tf_computation(reservoir_type, reservoir_type)
  @tf.function
  def merge_samples(a, b):
    # First concatenate all the values together. If the size of the resulting
    # structure is less than the sample size we don't need to do anything else.
    merged_random_values = tf.concat([a['random_values'], b['random_values']],
                                     axis=0)
    merged_samples = tf.nest.map_structure(
        lambda x, y: tf.concat([x, y], axis=0), a['samples'], b['samples'])
    if tf.size(merged_random_values) <= sample_size:
      return collections.OrderedDict(
          # `random_seed` is no longer used, but we need to keep the structure
          # for this reduction method. Arbitrarily forward the seed from `a`.
          random_seed=a['random_seed'],
          random_values=merged_random_values,
          samples=merged_samples)
    # Otherwise we need to select just the top values based on sample size.
    _, indices = tf.nn.top_k(merged_random_values, sample_size, sorted=False)
    selection_mask = tf.scatter_nd(
        indices=tf.expand_dims(indices, axis=-1),
        updates=tf.fill(dims=tf.shape(indices), value=True),
        shape=tf.shape(merged_random_values))
    selected_random_values = tf.boolean_mask(
        merged_random_values, mask=selection_mask)
    selected_samples = tf.nest.map_structure(
        lambda t: tf.boolean_mask(t, mask=selection_mask), merged_samples)
    return collections.OrderedDict(
        # `random_seed` is no longer used, but we need to keep the structure
        # (the type) for this reduction method. Arbitrarily forward the seed
        # from `a`.
        random_seed=a['random_seed'],
        random_values=selected_random_values,
        samples=selected_samples)

  return merge_samples


def _build_finalize_sample(
    value_type: computation_types.Type) -> computation_base.Computation:
  """Builds the `report` computation for sampling."""
  reservoir_type = _build_reservoir_type(value_type)

  @computations.tf_computation(reservoir_type)
  @tf.function
  def finalize_samples(reservoir):
    # Drop all the container extra data and just return the sampled values.
    return reservoir['samples']

  return finalize_samples


class UnweightedReservoirSamplingFactory(factory.UnweightedAggregationFactory):
  """An `UnweightedAggregationFactory` for reservoir sampling values.

  The created `tff.templates.AggregationProcess` samples values placed at
  `CLIENTS`, and outputs the sample placed at `SERVER`.

  The process has empty `state`. The `measurements` of this factory include
  the number of non-finite (`NaN` or `Inf` values) for each leaf in the value
  structure.
  """

  def __init__(self, sample_size: int):
    py_typecheck.check_type(sample_size, int)
    if sample_size <= 0:
      raise ValueError('`sample_size` must be positive.')
    self._sample_size = sample_size

  def create(
      self,
      value_type: factory.ValueType) -> aggregation_process.AggregationProcess:

    @computations.federated_computation()
    def init_fn():
      # Empty/null state, nothing is tracked across invocations.
      return intrinsics.federated_value((), placements.SERVER)

    @computations.federated_computation(
        computation_types.at_server(()),
        computation_types.at_clients(value_type))
    def next_fn(unused_state, value):
      # Empty tuple is the `None` of TFF.
      empty_tuple = intrinsics.federated_value((), placements.SERVER)
      value_type = value.type_signature.member
      initial_reservoir = _build_initial_sample_reservoir(value_type)
      sample_value = _build_sample_value(value_type, self._sample_size)
      merge_samples = _build_merge_samples(value_type, self._sample_size)
      finalize_sample = _build_finalize_sample(value_type)
      samples = intrinsics.federated_aggregate(
          value,
          zero=initial_reservoir,
          accumulate=sample_value,
          merge=merge_samples,
          report=finalize_sample)
      return measured_process.MeasuredProcessOutput(
          state=empty_tuple, result=samples, measurements=empty_tuple)

    return aggregation_process.AggregationProcess(init_fn, next_fn)
