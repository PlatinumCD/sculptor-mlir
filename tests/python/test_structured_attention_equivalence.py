#!/usr/bin/env python3

import math

import numpy as np


def scalar_attention(query, key, value, num_heads, causal):
    batch_size, query_length, hidden_size = query.shape
    key_length = key.shape[1]
    head_dim = hidden_size // num_heads
    scores = np.empty(
        (batch_size, num_heads, query_length, key_length), dtype=np.float32
    )

    for batch in range(batch_size):
        for head in range(num_heads):
            for query_index in range(query_length):
                for key_index in range(key_length):
                    score = np.float32(0.0)
                    for dim in range(head_dim):
                        feature = head * head_dim + dim
                        score += query[batch, query_index, feature] * key[
                            batch, key_index, feature
                        ]
                    if causal and key_index > query_index:
                        score = np.float32(-np.inf)
                    else:
                        score *= np.float32(1.0 / math.sqrt(head_dim))
                    scores[batch, head, query_index, key_index] = score

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    heads = np.zeros(
        (batch_size, num_heads, query_length, head_dim), dtype=np.float32
    )
    for batch in range(batch_size):
        for head in range(num_heads):
            for query_index in range(query_length):
                for dim in range(head_dim):
                    feature = head * head_dim + dim
                    for key_index in range(key_length):
                        heads[batch, head, query_index, dim] += probabilities[
                            batch, head, query_index, key_index
                        ] * value[batch, key_index, feature]

    recombined = np.empty(
        (batch_size, query_length, hidden_size), dtype=np.float32
    )
    for batch in range(batch_size):
        for head in range(num_heads):
            for query_index in range(query_length):
                for dim in range(head_dim):
                    recombined[
                        batch, query_index, head * head_dim + dim
                    ] = heads[batch, head, query_index, dim]
    return scores, probabilities, heads, recombined


def structured_attention(query, key, value, num_heads, causal):
    batch_size, query_length, hidden_size = query.shape
    key_length = key.shape[1]
    head_dim = hidden_size // num_heads
    query_heads = query.reshape(batch_size, query_length, num_heads, head_dim)
    key_heads = key.reshape(batch_size, key_length, num_heads, head_dim)
    value_heads = value.reshape(batch_size, key_length, num_heads, head_dim)

    scores = np.einsum("bqhd,bkhd->bhqk", query_heads, key_heads, dtype=np.float32)
    scores *= np.float32(1.0 / math.sqrt(head_dim))
    if causal:
        query_indices = np.arange(query_length)[:, None]
        key_indices = np.arange(key_length)[None, :]
        scores = np.where(key_indices > query_indices, -np.inf, scores)

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
    heads = np.einsum(
        "bhqk,bkhd->bhqd", probabilities, value_heads, dtype=np.float32
    )
    recombined = heads.transpose(0, 2, 1, 3).reshape(
        batch_size, query_length, hidden_size
    )
    return scores, probabilities, heads, recombined


def check_case(query_length, key_length, causal):
    rng = np.random.default_rng(7 + query_length + key_length + int(causal))
    shape = (2, query_length, 6)
    query = rng.standard_normal(shape, dtype=np.float32)
    key = rng.standard_normal((2, key_length, 6), dtype=np.float32)
    value = rng.standard_normal((2, key_length, 6), dtype=np.float32)
    scalar = scalar_attention(query, key, value, num_heads=3, causal=causal)
    structured = structured_attention(
        query, key, value, num_heads=3, causal=causal
    )
    for expected, actual in zip(scalar, structured):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def main():
    check_case(query_length=4, key_length=4, causal=False)
    check_case(query_length=4, key_length=4, causal=True)
    check_case(query_length=3, key_length=5, causal=False)
    print("structured attention matches the scalar reference")


if __name__ == "__main__":
    main()
