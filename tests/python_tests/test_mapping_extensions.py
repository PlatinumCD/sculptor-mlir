#!/usr/bin/env python3
"""Focused regressions for optional mapping extensions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import torch

from lowering_harness import (
    ANALOG_ROOT,
    DEFAULT_SCULPTOR_OPT,
    LoweringCase,
    export_linalg,
    initialize_parameters,
)


THIS_DIR = Path(__file__).resolve().parent
COST_PROFILE = THIS_DIR / "data" / "calibrated_mapping_costs.json"


DIGITAL_CHAIN_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty0 = tensor.empty() : tensor<1024xf32>
    %first = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<1024xf32>) outs(%empty0 : tensor<1024xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1024xf32>
    %empty1 = tensor.empty() : tensor<1024xf32>
    %second = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%first : tensor<1024xf32>) outs(%empty1 : tensor<1024xf32>) {
    ^bb0(%input: f32, %output: f32):
      %two = arith.constant 2.0 : f32
      %value = arith.mulf %input, %two : f32
      linalg.yield %value : f32
    } -> tensor<1024xf32>
    return %second : tensor<1024xf32>
  }
}
"""


SAME_LAYER_DIGITAL_CHAIN_FIXTURE = DIGITAL_CHAIN_FIXTURE.replace(
    'iterator_types = ["parallel"]',
    'iterator_types = ["parallel"],\n'
    '      sculptor.semantic.layer_id = 7 : i64,\n'
    '      sculptor.semantic.layer_kind = "activation"',
)


FAN_OUT_DIGITAL_CHAIN_FIXTURE = DIGITAL_CHAIN_FIXTURE.replace(
    '-> tensor<1024xf32> {',
    '-> (tensor<1024xf32>, tensor<1024xf32>) {',
    1,
).replace(
    'return %second : tensor<1024xf32>',
    'return %first, %second : tensor<1024xf32>, tensor<1024xf32>',
)


STATIC_NEAREST_2X_FIXTURE = r"""
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x4x4xf32> {
    %c0 = arith.constant 0 : index
    %scale = arith.constant 2.0 : f32
    %empty = tensor.empty() : tensor<1x2x4x4xf32>
    %resized = linalg.generic {
      indexing_maps = [#identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%empty : tensor<1x2x4x4xf32>) {
    ^bb0(%out: f32):
      %channel = linalg.index 1 : index
      %row = linalg.index 2 : index
      %column = linalg.index 3 : index
      %row_i64 = arith.index_cast %row : index to i64
      %row_f32 = arith.sitofp %row_i64 : i64 to f32
      %row_scaled = arith.divf %row_f32, %scale : f32
      %row_floor = math.floor %row_scaled : f32
      %row_result_i64 = arith.fptosi %row_floor : f32 to i64
      %row_result = arith.index_cast %row_result_i64 : i64 to index
      %column_i64 = arith.index_cast %column : index to i64
      %column_f32 = arith.sitofp %column_i64 : i64 to f32
      %column_scaled = arith.divf %column_f32, %scale : f32
      %column_floor = math.floor %column_scaled : f32
      %column_result_i64 = arith.fptosi %column_floor : f32 to i64
      %column_result = arith.index_cast %column_result_i64 : i64 to index
      %value = tensor.extract %arg0[
        %c0, %channel, %row_result, %column_result
      ] : tensor<1x2x2x2xf32>
      linalg.yield %value : f32
    } -> tensor<1x2x4x4xf32>
    return %resized : tensor<1x2x4x4xf32>
  }
}
"""


STATIC_NEAREST_IDENTITY_FIXTURE = r"""
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
    %c0 = arith.constant 0 : index
    %empty = tensor.empty() : tensor<1x2x2x2xf32>
    %resized = linalg.generic {
      indexing_maps = [#identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } outs(%empty : tensor<1x2x2x2xf32>) {
    ^bb0(%out: f32):
      %channel = linalg.index 1 : index
      %row = linalg.index 2 : index
      %column = linalg.index 3 : index
      %row_i64 = arith.index_cast %row : index to i64
      %row_f32 = arith.sitofp %row_i64 : i64 to f32
      %row_floor = math.floor %row_f32 : f32
      %row_result_i64 = arith.fptosi %row_floor : f32 to i64
      %row_result = arith.index_cast %row_result_i64 : i64 to index
      %column_i64 = arith.index_cast %column : index to i64
      %column_f32 = arith.sitofp %column_i64 : i64 to f32
      %column_floor = math.floor %column_f32 : f32
      %column_result_i64 = arith.fptosi %column_floor : f32 to i64
      %column_result = arith.index_cast %column_result_i64 : i64 to index
      %value = tensor.extract %arg0[
        %c0, %channel, %row_result, %column_result
      ] : tensor<1x2x2x2xf32>
      linalg.yield %value : f32
    } -> tensor<1x2x2x2xf32>
    return %resized : tensor<1x2x2x2xf32>
  }
}
"""


LAYER_EPILOGUE_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %empty0 = tensor.empty() : tensor<16xf32>
    %layer = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"],
      sculptor.semantic.layer_id = 7 : i64,
      sculptor.semantic.layer_kind = "conv2d"
    } ins(%arg0 : tensor<16xf32>) outs(%empty0 : tensor<16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %two = arith.constant 2.0 : f32
      %value = arith.mulf %input, %two : f32
      linalg.yield %value : f32
    } -> tensor<16xf32>
    %empty1 = tensor.empty() : tensor<16xf32>
    %relu = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%layer : tensor<16xf32>) outs(%empty1 : tensor<16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %zero = arith.constant 0.0 : f32
      %value = arith.maximumf %input, %zero : f32
      linalg.yield %value : f32
    } -> tensor<16xf32>
    return %relu : tensor<16xf32>
  }
}
"""


AMBIGUOUS_LAYER_JOIN_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<16xf32>, %arg1: tensor<16xf32>)
      -> tensor<16xf32> {
    %empty0 = tensor.empty() : tensor<16xf32>
    %left = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"],
      sculptor.semantic.layer_id = 7 : i64,
      sculptor.semantic.layer_kind = "conv2d"
    } ins(%arg0 : tensor<16xf32>) outs(%empty0 : tensor<16xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<16xf32>
    %empty1 = tensor.empty() : tensor<16xf32>
    %right = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"],
      sculptor.semantic.layer_id = 8 : i64,
      sculptor.semantic.layer_kind = "conv2d"
    } ins(%arg1 : tensor<16xf32>) outs(%empty1 : tensor<16xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<16xf32>
    %empty2 = tensor.empty() : tensor<16xf32>
    %join = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]
    } ins(%left, %right : tensor<16xf32>, tensor<16xf32>)
      outs(%empty2 : tensor<16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %output: f32):
      %value = arith.addf %lhs, %rhs : f32
      linalg.yield %value : f32
    } -> tensor<16xf32>
    return %join : tensor<16xf32>
  }
}
"""


READ_BEFORE_WRITE_OUT_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %empty = tensor.empty() : tensor<16xf32>
    %first = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"],
      sculptor.semantic.layer_id = 7 : i64,
      sculptor.semantic.layer_kind = "producer"
    } ins(%arg0 : tensor<16xf32>) outs(%empty : tensor<16xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<16xf32>
    %second = linalg.generic {
      indexing_maps = [#identity],
      iterator_types = ["parallel"],
      sculptor.semantic.layer_id = 8 : i64,
      sculptor.semantic.layer_kind = "accumulator"
    } outs(%first : tensor<16xf32>) {
    ^bb0(%output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %output, %one : f32
      linalg.yield %value : f32
    } -> tensor<16xf32>
    return %second : tensor<16xf32>
  }
}
"""


DIGITAL_FORK_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<1024xf32>,
                     %arg1: tensor<1024xf32>)
      -> (tensor<1024xf32>, tensor<1024xf32>) {
    %empty0 = tensor.empty() : tensor<1024xf32>
    %left = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<1024xf32>) outs(%empty0 : tensor<1024xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1024xf32>
    %empty1 = tensor.empty() : tensor<1024xf32>
    %right = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%arg1 : tensor<1024xf32>) outs(%empty1 : tensor<1024xf32>) {
    ^bb0(%input: f32, %output: f32):
      %two = arith.constant 2.0 : f32
      %value = arith.mulf %input, %two : f32
      linalg.yield %value : f32
    } -> tensor<1024xf32>
    return %left, %right : tensor<1024xf32>, tensor<1024xf32>
  }
}
"""


SEGMENTED_TRANSPOSE_FIXTURE = r"""
#input = affine_map<(d0, d1) -> (d0, d1)>
#output = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @forward(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %empty = tensor.empty() : tensor<8x8xf32>
    %result = linalg.generic {
      indexing_maps = [#input, #output],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>) {
    ^bb0(%input_value: f32, %output_value: f32):
      linalg.yield %input_value : f32
    } -> tensor<8x8xf32>
    return %result : tensor<8x8xf32>
  }
}
"""


PACKED_TRANSPOSE_FIXTURE = r"""
#input = affine_map<(d0, d1) -> (d0, d1)>
#output = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @forward(
      %arg0: tensor<2x4097xf32>) -> tensor<4097x2xf32> {
    %empty = tensor.empty() : tensor<4097x2xf32>
    %result = linalg.generic {
      indexing_maps = [#input, #output],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<2x4097xf32>)
      outs(%empty : tensor<4097x2xf32>) {
    ^bb0(%input_value: f32, %output_value: f32):
      linalg.yield %input_value : f32
    } -> tensor<4097x2xf32>
    return %result : tensor<4097x2xf32>
  }
}
"""


ASSEMBLED_TRANSPOSE_FIXTURE = r"""
module {
  func.func @forward(%lhs: tensor<1x3x4x4xf32>,
                     %rhs: tensor<1x3x4x4xf32>)
      -> tensor<1x2x4x2x3xf32> {
    %assembled_empty = tensor.empty() : tensor<1x6x4x4xf32>
    %with_lhs = tensor.insert_slice %lhs into %assembled_empty
      [0, 0, 0, 0] [1, 3, 4, 4] [1, 1, 1, 1]
      : tensor<1x3x4x4xf32> into tensor<1x6x4x4xf32>
    %assembled = tensor.insert_slice %rhs into %with_lhs
      [0, 3, 0, 0] [1, 3, 4, 4] [1, 1, 1, 1]
      : tensor<1x3x4x4xf32> into tensor<1x6x4x4xf32>
    %expanded = tensor.expand_shape %assembled [[0], [1, 2], [3], [4]]
      output_shape [1, 2, 3, 4, 4]
      : tensor<1x6x4x4xf32> into tensor<1x2x3x4x4xf32>
    %slice = tensor.extract_slice %expanded[0, 0, 0, 0, 0]
      [1, 2, 3, 2, 4] [1, 1, 1, 1, 1]
      : tensor<1x2x3x4x4xf32> to tensor<1x2x3x2x4xf32>
    %output = tensor.empty() : tensor<1x2x4x2x3xf32>
    %transposed = linalg.transpose
      ins(%slice : tensor<1x2x3x2x4xf32>)
      outs(%output : tensor<1x2x4x2x3xf32>)
      permutation = [0, 3, 4, 1, 2]
    return %transposed : tensor<1x2x4x2x3xf32>
  }
}
"""


NESTED_ASSEMBLY_FIXTURE = r"""
#identity4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%a: tensor<1x2x2x3xf32>,
                     %b: tensor<1x2x2x3xf32>,
                     %c: tensor<1x2x2x3xf32>,
                     %d: tensor<1x2x2x3xf32>) -> tensor<1x16x3xf32> {
    %ea = tensor.empty() : tensor<1x2x2x3xf32>
    %va = linalg.generic {
      indexing_maps = [#identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%a : tensor<1x2x2x3xf32>)
      outs(%ea : tensor<1x2x2x3xf32>) {
    ^bb0(%x: f32, %out: f32):
      %one = arith.constant 1.0 : f32
      %y = arith.addf %x, %one : f32
      linalg.yield %y : f32
    } -> tensor<1x2x2x3xf32>
    %eb = tensor.empty() : tensor<1x2x2x3xf32>
    %vb = linalg.generic {
      indexing_maps = [#identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%b : tensor<1x2x2x3xf32>)
      outs(%eb : tensor<1x2x2x3xf32>) {
    ^bb0(%x: f32, %out: f32):
      %two = arith.constant 2.0 : f32
      %y = arith.addf %x, %two : f32
      linalg.yield %y : f32
    } -> tensor<1x2x2x3xf32>
    %ec = tensor.empty() : tensor<1x2x2x3xf32>
    %vc = linalg.generic {
      indexing_maps = [#identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%c : tensor<1x2x2x3xf32>)
      outs(%ec : tensor<1x2x2x3xf32>) {
    ^bb0(%x: f32, %out: f32):
      %three = arith.constant 3.0 : f32
      %y = arith.addf %x, %three : f32
      linalg.yield %y : f32
    } -> tensor<1x2x2x3xf32>
    %ed = tensor.empty() : tensor<1x2x2x3xf32>
    %vd = linalg.generic {
      indexing_maps = [#identity4, #identity4],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%d : tensor<1x2x2x3xf32>)
      outs(%ed : tensor<1x2x2x3xf32>) {
    ^bb0(%x: f32, %out: f32):
      %four = arith.constant 4.0 : f32
      %y = arith.addf %x, %four : f32
      linalg.yield %y : f32
    } -> tensor<1x2x2x3xf32>
    %left_empty = tensor.empty() : tensor<1x4x2x3xf32>
    %left_a = tensor.insert_slice %va into %left_empty
      [0, 0, 0, 0] [1, 2, 2, 3] [1, 1, 1, 1]
      : tensor<1x2x2x3xf32> into tensor<1x4x2x3xf32>
    %left = tensor.insert_slice %vb into %left_a
      [0, 2, 0, 0] [1, 2, 2, 3] [1, 1, 1, 1]
      : tensor<1x2x2x3xf32> into tensor<1x4x2x3xf32>
    %left_flat = tensor.collapse_shape %left [[0], [1, 2], [3]]
      : tensor<1x4x2x3xf32> into tensor<1x8x3xf32>
    %right_empty = tensor.empty() : tensor<1x4x2x3xf32>
    %right_c = tensor.insert_slice %vc into %right_empty
      [0, 0, 0, 0] [1, 2, 2, 3] [1, 1, 1, 1]
      : tensor<1x2x2x3xf32> into tensor<1x4x2x3xf32>
    %right = tensor.insert_slice %vd into %right_c
      [0, 2, 0, 0] [1, 2, 2, 3] [1, 1, 1, 1]
      : tensor<1x2x2x3xf32> into tensor<1x4x2x3xf32>
    %right_flat = tensor.collapse_shape %right [[0], [1, 2], [3]]
      : tensor<1x4x2x3xf32> into tensor<1x8x3xf32>
    %output_empty = tensor.empty() : tensor<1x16x3xf32>
    %output_left = tensor.insert_slice %left_flat into %output_empty
      [0, 0, 0] [1, 8, 3] [1, 1, 1]
      : tensor<1x8x3xf32> into tensor<1x16x3xf32>
    %output = tensor.insert_slice %right_flat into %output_left
      [0, 8, 0] [1, 8, 3] [1, 1, 1]
      : tensor<1x8x3xf32> into tensor<1x16x3xf32>
    return %output : tensor<1x16x3xf32>
  }
}
"""


STATIC_PAD_FIXTURE = r"""
module {
  func.func @forward(%input: tensor<1x4xf32>) -> tensor<1x6xf32> {
    %padded = tensor.pad %input low[0, 1] high[0, 1] {
    ^bb0(%row: index, %column: index):
      %zero = arith.constant 0.0 : f32
      tensor.yield %zero : f32
    } : tensor<1x4xf32> to tensor<1x6xf32>
    return %padded : tensor<1x6xf32>
  }
}
"""


STATIC_VIEW_CHAIN_PAD_FIXTURE = r"""
#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @forward(%lhs: tensor<1x8x4x4xf32>,
                     %rhs: tensor<1x8x4x4xf32>)
      -> tensor<1x4x6x6xf32> {
    %lhs_empty = tensor.empty() : tensor<1x8x4x4xf32>
    %lhs_value = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%lhs : tensor<1x8x4x4xf32>)
      outs(%lhs_empty : tensor<1x8x4x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x8x4x4xf32>
    %rhs_empty = tensor.empty() : tensor<1x8x4x4xf32>
    %rhs_value = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%rhs : tensor<1x8x4x4xf32>)
      outs(%rhs_empty : tensor<1x8x4x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      %two = arith.constant 2.0 : f32
      %value = arith.addf %input, %two : f32
      linalg.yield %value : f32
    } -> tensor<1x8x4x4xf32>
    %assembled_empty = tensor.empty() : tensor<1x16x4x4xf32>
    %with_lhs = tensor.insert_slice %lhs_value into %assembled_empty
      [0, 0, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
      : tensor<1x8x4x4xf32> into tensor<1x16x4x4xf32>
    %assembled = tensor.insert_slice %rhs_value into %with_lhs
      [0, 8, 0, 0] [1, 8, 4, 4] [1, 1, 1, 1]
      : tensor<1x8x4x4xf32> into tensor<1x16x4x4xf32>
    %flat = tensor.collapse_shape %assembled [[0, 1, 2, 3]]
      : tensor<1x16x4x4xf32> into tensor<256xf32>
    %second_half = tensor.extract_slice %flat[128] [128] [1]
      : tensor<256xf32> to tensor<128xf32>
    %expanded = tensor.expand_shape %second_half [[0, 1, 2, 3]]
      output_shape [1, 8, 4, 4]
      : tensor<128xf32> into tensor<1x8x4x4xf32>
    %channels = tensor.extract_slice %expanded[0, 4, 0, 0] [1, 4, 4, 4]
      [1, 1, 1, 1]
      : tensor<1x8x4x4xf32> to tensor<1x4x4x4xf32>
    %padded = tensor.pad %channels low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%batch: index, %channel: index, %row: index, %column: index):
      %zero = arith.constant 0.0 : f32
      tensor.yield %zero : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    return %padded : tensor<1x4x6x6xf32>
  }
}
"""


RETURNED_AND_REUSED_OUTPUT_FIXTURE = r"""
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%arg0: tensor<8x8xf32>)
      -> (tensor<8x8xf32>, tensor<1x8xf32>) {
    %empty = tensor.empty() : tensor<8x8xf32>
    %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<8x8xf32>) outs(%empty : tensor<8x8xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<8x8xf32>
    %slice = tensor.extract_slice %result[0, 0] [1, 8] [1, 1]
      : tensor<8x8xf32> to tensor<1x8xf32>
    return %result, %slice : tensor<8x8xf32>, tensor<1x8xf32>
  }
}
"""


INLINE_SPLAT_MVM_FIXTURE = r"""
module {
  func.func @forward(%input: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %matrix = arith.constant dense<1.0> : tensor<4x4xf32>
    %result = sculptor.mvm %input, %matrix
      : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
    return %result : tensor<1x4xf32>
  }
}
"""


CONTIGUOUS_GATHER_FIXTURE = r"""
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%source: tensor<8xf32>) -> tensor<4xf32> {
    %index_empty = tensor.empty() : tensor<4xi64>
    %indices = linalg.generic {
      indexing_maps = [#identity], iterator_types = ["parallel"]
    } outs(%index_empty : tensor<4xi64>) {
    ^bb0(%out: i64):
      %index = linalg.index 0 : index
      %index_i64 = arith.index_cast %index : index to i64
      %offset = arith.constant 4 : i64
      %shifted = arith.addi %index_i64, %offset : i64
      linalg.yield %shifted : i64
    } -> tensor<4xi64>
    %output_empty = tensor.empty() : tensor<4xf32>
    %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]
    } ins(%indices : tensor<4xi64>) outs(%output_empty : tensor<4xf32>) {
    ^bb0(%index: i64, %out: f32):
      %zero = arith.constant 0 : i64
      %extent = arith.constant 8 : i64
      %negative = arith.cmpi slt, %index, %zero : i64
      %wrapped = arith.addi %index, %extent : i64
      %normalized = arith.select %negative, %wrapped, %index : i64
      %position = arith.index_cast %normalized : i64 to index
      %value = tensor.extract %source[%position] : tensor<8xf32>
      linalg.yield %value : f32
    } -> tensor<4xf32>
    return %result : tensor<4xf32>
  }
}
"""


LIFETIME_CHAIN_FIXTURE = r"""
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %empty0 = tensor.empty() : tensor<1x16xf32>
    %first = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<1x16xf32>) outs(%empty0 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty1 = tensor.empty() : tensor<1x16xf32>
    %second = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%first : tensor<1x16xf32>) outs(%empty1 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty2 = tensor.empty() : tensor<1x16xf32>
    %third = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%second : tensor<1x16xf32>) outs(%empty2 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty3 = tensor.empty() : tensor<1x16xf32>
    %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%third : tensor<1x16xf32>) outs(%empty3 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    return %result : tensor<1x16xf32>
  }
}
"""


LIFETIME_FANOUT_FIXTURE = r"""
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%arg0: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %empty0 = tensor.empty() : tensor<1x16xf32>
    %shared = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<1x16xf32>) outs(%empty0 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty1 = tensor.empty() : tensor<1x16xf32>
    %left = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%shared : tensor<1x16xf32>) outs(%empty1 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %one = arith.constant 1.0 : f32
      %value = arith.addf %input, %one : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty2 = tensor.empty() : tensor<1x16xf32>
    %right = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%shared : tensor<1x16xf32>) outs(%empty2 : tensor<1x16xf32>) {
    ^bb0(%input: f32, %output: f32):
      %two = arith.constant 2.0 : f32
      %value = arith.addf %input, %two : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    %empty3 = tensor.empty() : tensor<1x16xf32>
    %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%left, %right : tensor<1x16xf32>, tensor<1x16xf32>)
      outs(%empty3 : tensor<1x16xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %output: f32):
      %value = arith.addf %lhs, %rhs : f32
      linalg.yield %value : f32
    } -> tensor<1x16xf32>
    return %result : tensor<1x16xf32>
  }
}
"""


CONCAT_PRODUCER_CONTRIBUTION_FIXTURE = r"""
#identity = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%lhs: tensor<2x4xf32>, %rhs: tensor<2x4xf32>)
      -> tensor<4x4xf32> {
    %lhs_empty = tensor.empty() : tensor<2x4xf32>
    %lhs_value = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%lhs : tensor<2x4xf32>) outs(%lhs_empty : tensor<2x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<2x4xf32>
    %rhs_empty = tensor.empty() : tensor<2x4xf32>
    %rhs_value = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%rhs : tensor<2x4xf32>) outs(%rhs_empty : tensor<2x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<2x4xf32>
    %joined = tensor.concat dim(0) %lhs_value, %rhs_value
      : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
    %output_empty = tensor.empty() : tensor<4x4xf32>
    %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%joined : tensor<4x4xf32>)
      outs(%output_empty : tensor<4x4xf32>) {
    ^bb0(%input: f32, %output: f32):
      linalg.yield %input : f32
    } -> tensor<4x4xf32>
    return %result : tensor<4x4xf32>
  }
}
"""


SHARED_DESTINATION_INITIALIZER_FIXTURE = r"""
module {
  func.func @forward(
      %lhs0: tensor<2x3xf32>, %rhs0: tensor<3x2xf32>,
      %lhs1: tensor<2x3xf32>, %rhs1: tensor<3x2xf32>)
      -> (tensor<2x2xf32>, tensor<2x2xf32>) {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<2x2xf32>
    %initializer = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    %left = linalg.matmul
      ins(%lhs0, %rhs0 : tensor<2x3xf32>, tensor<3x2xf32>)
      outs(%initializer : tensor<2x2xf32>) -> tensor<2x2xf32>
    %right = linalg.matmul
      ins(%lhs1, %rhs1 : tensor<2x3xf32>, tensor<3x2xf32>)
      outs(%initializer : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %left, %right : tensor<2x2xf32>, tensor<2x2xf32>
  }
}
"""


class FourWayReductionLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


class FullAnalogOccupancyWithDigitalTail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return torch.relu(self.linear(value) + 1.0)


class ThreeWayReductionLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.linear(value)


class TwoLayerFanOutLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 8, bias=False)
        self.second = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.second(self.first(value))


class ThreeLayerRouteLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(8, 8, bias=False)
        self.second = torch.nn.Linear(8, 8, bias=False)
        self.third = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, value):
        return self.third(self.second(self.first(value)))


class ParallelLinearBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(8, 8, bias=False)
        self.right = torch.nn.Linear(8, 8, bias=False)
        initialize_parameters(self)

    def forward(self, left, right):
        return self.left(left), self.right(right)


class SixLayerRouteLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8, bias=False) for _ in range(6)]
        )
        initialize_parameters(self)

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


class ElementwiseChain(torch.nn.Module):
    def forward(self, left, right):
        summed = left + right
        return torch.relu(summed)


class LongElementwiseChain(torch.nn.Module):
    def forward(self, value):
        first = torch.relu(value + 1.0)
        second = first * 2.0
        return torch.relu(second + 3.0)


class ForkedElementwiseChain(torch.nn.Module):
    def forward(self, value):
        left = torch.relu(value + 1.0)
        right = torch.relu(value + 2.0)
        return left + right


class SharedElementwiseFanOut(torch.nn.Module):
    def forward(self, value):
        shared = torch.relu(value + 1.0)
        left = shared + 2.0
        right = shared + 3.0
        return left + right


class TransposeSandwich(torch.nn.Module):
    def forward(self, value):
        produced = value + 1.0
        transposed = produced.transpose(0, 1).contiguous()
        return transposed + 2.0


class LayerNormOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, value):
        return self.norm(value)


class DynamicMatmulJoin(torch.nn.Module):
    def forward(self, left, right):
        query = torch.relu(left + 1.0)
        key = torch.relu(right + 1.0)
        return torch.matmul(query, key)


class FrozenBatchNormConv(torch.nn.Module):
    """RetinaNet-style convolution followed by frozen batch normalization."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
        self.register_buffer("weight", torch.tensor([1.25, 0.75, 1.5]))
        self.register_buffer("bias", torch.tensor([0.5, -0.25, 0.125]))
        self.register_buffer("running_mean", torch.tensor([0.2, -0.4, 0.6]))
        self.register_buffer("running_var", torch.tensor([0.5, 1.5, 2.5]))

    def forward(self, value):
        result = self.conv(value)
        scale = self.weight * torch.rsqrt(self.running_var + 1.0e-5)
        bias = self.bias - self.running_mean * scale
        return result * scale.reshape(1, -1, 1, 1) + bias.reshape(
            1, -1, 1, 1
        )


def run_pipeline(case: LoweringCase, passes: list[str]) -> str:
    result = subprocess.run(
        [str(DEFAULT_SCULPTOR_OPT), "-", *passes],
        input=export_linalg(case),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"{case.name} failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )
    return result.stdout


def run_ir_pipeline(module: str, passes: list[str]) -> str:
    result = subprocess.run(
        [str(DEFAULT_SCULPTOR_OPT), "-", *passes],
        input=module,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise AssertionError(
            f"MLIR fixture failed with exit code {result.returncode}:\n"
            f"{result.stderr}"
        )
    return result.stdout


def run_serialized_pipeline(
    case: LoweringCase, producing_passes: list[str], consuming_passes: list[str]
) -> str:
    produced = run_pipeline(case, producing_passes)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "placed.mlir"
        path.write_text(produced)
        result = subprocess.run(
            [str(DEFAULT_SCULPTOR_OPT), str(path), *consuming_passes],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise AssertionError(
            f"{case.name} failed after serialization with exit code "
            f"{result.returncode}:\n{result.stderr}"
        )
    return result.stdout


def build_ra_tree_report(module: str) -> dict:
    report_tool = DEFAULT_SCULPTOR_OPT.parent / "sculptor-ra-tree-report"
    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "placed.mlir"
        json_path = Path(directory) / "report.json"
        html_path = Path(directory) / "report.html"
        input_path.write_text(module)
        result = subprocess.run(
            [
                str(report_tool),
                str(input_path),
                f"--json-output={json_path}",
                "-o",
                str(html_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(f"RA-tree report failed:\n{result.stderr}")
        return json.loads(json_path.read_text())


def linear_prefix(array_cols: int = 2) -> list[str]:
    return [
        "--sculptor-canonicalize-layers",
        "--sculptor-extract-layers",
        "--sculptor-convert-layers",
        (
            "--sculptor-expand-mvm-to-golem="
            f"array-rows=8 array-cols={array_cols}"
        ),
    ]


def materialize_outlined_tiles(outlined: str) -> list[str]:
    tile_ids = sorted(
        {int(value) for value in re.findall(r"module @tile_(\d+)", outlined)}
    )
    if not tile_ids:
        raise AssertionError("outlined deployment contains no tile modules")
    materialized = []
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outlined.mlir"
        path.write_text(outlined)
        for tile_id in tile_ids:
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    str(path),
                    f"--sculptor-extract-tile-module=tile-id={tile_id}",
                    "--sculptor-materialize-tile-runtime-graph",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                raise AssertionError(
                    f"tile {tile_id} runtime materialization failed:\n"
                    f"{result.stderr}"
                )
            materialized.append(result.stdout)
    return materialized


def emit_outlined_tile_abi(
    outlined: str,
    tile_id: int,
    pre_finalize_passes: tuple[str, ...] = (),
    strict_memory_audit: bool = True,
) -> str:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "outlined.mlir"
        path.write_text(outlined)
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                str(path),
                f"--sculptor-extract-tile-module=tile-id={tile_id}",
                "--sculptor-materialize-tile-runtime-graph",
                *pre_finalize_passes,
                "--sculptor-finalize-tile-runtime-graph",
                "--sculptor-lower-golem-to-llvm-shims",
                "--canonicalize",
                "--cse",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map",
                "--buffer-results-to-out-params=hoist-static-allocs",
                "--convert-bufferization-to-memref",
                "--sculptor-bind-tile-routine-destinations",
                "--buffer-hoisting",
                "--buffer-loop-hoisting",
                "--buffer-deallocation-pipeline",
                "--optimize-allocation-liveness",
                *(
                    ["--sculptor-audit-tile-bufferization=strict=true"]
                    if strict_memory_audit
                    else []
                ),
                "--sculptor-vectorize-tile-copies=vector-bits=256",
                "--convert-linalg-to-loops",
                "--lower-affine",
                "--convert-scf-to-cf",
                "--convert-vector-to-llvm",
                "--convert-math-to-libm",
                "--convert-math-to-llvm",
                "--expand-strided-metadata",
                "--lower-affine",
                "--convert-arith-to-llvm",
                "--convert-index-to-llvm",
                "--convert-cf-to-llvm",
                "--finalize-memref-to-llvm",
                "--convert-func-to-llvm",
                "--reconcile-unrealized-casts",
                "--sculptor-emit-golem-tile-abi",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise AssertionError(
            f"tile {tile_id} ABI emission failed:\n{result.stderr}"
        )
    return result.stdout


class MappingExtensionTest(unittest.TestCase):
    def test_elementwise_region_fusion_eliminates_single_use_boundary(self):
        fused = run_ir_pipeline(
            DIGITAL_CHAIN_FIXTURE,
            ["--sculptor-fuse-elementwise-regions", "--canonicalize", "--cse"],
        )
        self.assertEqual(fused.count("linalg.generic"), 1)
        self.assertIn("arith.addf", fused)
        self.assertIn("arith.mulf", fused)

    def test_elementwise_region_fusion_preserves_same_layer_identity(self):
        fused = run_ir_pipeline(
            SAME_LAYER_DIGITAL_CHAIN_FIXTURE,
            ["--sculptor-fuse-elementwise-regions", "--canonicalize", "--cse"],
        )
        self.assertEqual(fused.count("linalg.generic"), 1)
        self.assertEqual(fused.count("sculptor.semantic.layer_id = 7"), 1)
        self.assertEqual(
            fused.count('sculptor.semantic.layer_kind = "activation"'), 1
        )

    def test_elementwise_region_fusion_rejects_layer_boundary(self):
        fused = run_ir_pipeline(
            LAYER_EPILOGUE_FIXTURE,
            ["--sculptor-fuse-elementwise-regions", "--canonicalize", "--cse"],
        )
        self.assertEqual(fused.count("linalg.generic"), 2)
        self.assertEqual(fused.count("sculptor.semantic.layer_id = 7"), 1)

    def test_elementwise_region_fusion_rejects_shared_producer(self):
        fused = run_ir_pipeline(
            FAN_OUT_DIGITAL_CHAIN_FIXTURE,
            ["--sculptor-fuse-elementwise-regions", "--canonicalize", "--cse"],
        )
        self.assertEqual(fused.count("linalg.generic"), 2)

    def test_static_nearest_neighbor_uses_integer_indices(self):
        canonical = run_ir_pipeline(
            STATIC_NEAREST_2X_FIXTURE,
            ["--sculptor-canonicalize-layers", "--canonicalize", "--cse"],
        )
        self.assertNotIn("math.floor", canonical)
        self.assertNotIn("arith.sitofp", canonical)
        self.assertNotIn("arith.fptosi", canonical)
        self.assertNotIn("arith.divf", canonical)
        self.assertEqual(canonical.count("arith.divui"), 2)
        self.assertEqual(canonical.count("linalg.generic"), 1)

    def test_static_identity_nearest_neighbor_disappears(self):
        canonical = run_ir_pipeline(
            STATIC_NEAREST_IDENTITY_FIXTURE,
            ["--sculptor-canonicalize-layers", "--canonicalize", "--cse"],
        )
        self.assertNotIn("math.floor", canonical)
        self.assertNotIn("linalg.generic", canonical)
        self.assertRegex(
            canonical,
            r"return %arg0 : tensor<1x2x2x2xf32>",
        )

    def test_frozen_batch_norm_affine_folds_into_convolution(self):
        case = LoweringCase(
            name="frozen_batch_norm_conv",
            model_factory=FrozenBatchNormConv,
            input_factory=lambda: (torch.ones(1, 2, 4, 4),),
        )
        exported = export_linalg(case)
        before = run_ir_pipeline(
            exported,
            [
                "--sculptor-canonicalize-layers",
                "--canonicalize",
                "--cse",
            ],
        )
        after = run_ir_pipeline(
            exported,
            [
                "--sculptor-canonicalize-layers",
                "--sculptor-fold-inference-parameters",
                "--canonicalize",
                "--cse",
            ],
        )

        self.assertEqual(before.count("sculptor.nn.conv2d"), 1)
        self.assertIn("math.rsqrt", before)
        self.assertGreaterEqual(before.count("linalg.generic"), 5)
        self.assertEqual(after.count("sculptor.nn.conv2d"), 1)
        self.assertNotIn("math.rsqrt", after)
        self.assertNotIn("linalg.generic", after)
        self.assertIn("has_bias = true", after)
        self.assertIn("sculptor_folded_conv_weight_", after)
        self.assertRegex(after, r"dense<\[[^]]+\]> : tensor<3xf32>")

    def test_unambiguous_parallel_epilogue_inherits_semantic_layer_region(self):
        planned = run_ir_pipeline(
            LAYER_EPILOGUE_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping=strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        self.assertEqual(len(report["layer_regions"]), 1)
        region = report["layer_regions"][0]
        self.assertEqual(region["semantic_layer_id"], 7)
        self.assertEqual(region["semantic_layer_kind"], "conv2d")
        self.assertFalse(region["singleton_fallback"])
        self.assertEqual(len(region["operation_ids"]), 2)
        self.assertEqual(
            {operation["semantic_layer_id"] for operation in report["operations"]},
            {7},
        )

    def test_multi_layer_join_remains_a_fallback_region(self):
        planned = run_ir_pipeline(
            AMBIGUOUS_LAYER_JOIN_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping=strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        self.assertEqual(len(report["layer_regions"]), 3)
        fallback = [
            region
            for region in report["layer_regions"]
            if region["singleton_fallback"]
        ]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(len(fallback[0]["operation_ids"]), 1)

    def test_layer_cut_parallelizes_independent_parents_before_join(self):
        planned = run_ir_pipeline(
            AMBIGUOUS_LAYER_JOIN_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping=strategies=layer-cut "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        root = nodes[report["tree"]["root_id"]]
        self.assertEqual(root["kind"], "temporal_cut")
        parallel_parents = nodes[root["child_ids"][0]]
        self.assertEqual(parallel_parents["kind"], "spatial_cut")
        self.assertEqual(len(parallel_parents["child_ids"]), 2)
        self.assertEqual(nodes[root["child_ids"][1]]["kind"], "layer")
        self.assertEqual(
            sorted(len(region["input_tensors"])
                   for region in report["layer_regions"]),
            [1, 1, 2],
        )

    def test_recursive_fork_join_preserves_layer_cut_frontiers(self):
        planned = run_ir_pipeline(
            AMBIGUOUS_LAYER_JOIN_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=layer-cut,recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        root = nodes[report["tree"]["root_id"]]
        self.assertEqual(root["kind"], "temporal_cut")
        parallel_parents = nodes[root["child_ids"][0]]
        self.assertEqual(parallel_parents["kind"], "spatial_cut")
        self.assertEqual(len(parallel_parents["child_ids"]), 2)
        self.assertEqual(nodes[root["child_ids"][1]]["kind"], "layer")
        self.assertEqual(
            report["plan"]["planner"],
            "setup-first,layer-cut,recursive-fork-join",
        )

    def test_layer_cut_preserves_read_before_write_out_dependency(self):
        planned = run_ir_pipeline(
            READ_BEFORE_WRITE_OUT_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping=strategies=layer-cut "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        root = nodes[report["tree"]["root_id"]]
        self.assertEqual(root["kind"], "temporal_cut")
        self.assertEqual(
            [nodes[child]["kind"] for child in root["child_ids"]],
            ["layer", "layer"],
        )
        self.assertNotIn("spatial_cut", {node["kind"] for node in nodes.values()})
        accumulator = next(
            region
            for region in report["layer_regions"]
            if region["semantic_layer_id"] == 8
        )
        self.assertEqual(len(accumulator["input_tensors"]), 1)

    def test_earliest_finish_keeps_a_sequential_chain_local(self):
        def plan(policy: str) -> dict:
            planned = run_ir_pipeline(
                DIGITAL_CHAIN_FIXTURE,
                [
                    "--sculptor-build-ra-tree",
                    (
                        "--sculptor-plan-mapping="
                        "strategies=recursive-fork-join "
                        f"digital-scheduling-policy={policy} "
                        "mesh-rows=1 mesh-cols=2 arrays-per-core=1 "
                        "array-rows=8 array-cols=8 verify-plan"
                    ),
                ],
            )
            return build_ra_tree_report(planned)["functions"][0]

        balanced = plan("balanced")
        earliest = plan("earliest-finish")
        progressive = plan("progressive")

        def digital_tiles(report: dict) -> list[int]:
            return [
                assignment["tile_id"]
                for assignment in report["plan"]["realization"][
                    "leaf_assignments"
                ]
                if assignment["lane_kind"] == "digital"
            ]

        self.assertEqual(len(set(digital_tiles(balanced))), 2)
        self.assertEqual(len(set(digital_tiles(earliest))), 1)
        self.assertEqual(len(set(digital_tiles(progressive))), 1)
        self.assertGreater(len(balanced["logical_tile_graph"]["edges"]), 0)
        self.assertEqual(len(earliest["logical_tile_graph"]["edges"]), 0)
        self.assertEqual(len(progressive["logical_tile_graph"]["edges"]), 0)
        self.assertEqual(
            progressive["plan"]["digital_scheduling_policy"], "progressive"
        )

        assignments = earliest["plan"]["realization"]["leaf_assignments"]
        self.assertEqual(
            earliest["plan"]["estimated_latency_ns"],
            max(assignment["finish_ns"] for assignment in assignments),
        )

    def test_progressive_admission_opens_lanes_for_parallel_work(self):
        planned = run_ir_pipeline(
            DIGITAL_FORK_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "digital-scheduling-policy=progressive "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        assignments = [
            assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
            if assignment["lane_kind"] == "digital"
        ]
        self.assertEqual(len(assignments), 2)
        self.assertEqual(len({item["tile_id"] for item in assignments}), 2)
        self.assertEqual(
            report["plan"]["estimated_latency_ns"],
            max(item["finish_ns"] for item in assignments),
        )

    def test_sliding_window_retires_digital_tiles_without_revisiting(self):
        planned = run_ir_pipeline(
            DIGITAL_CHAIN_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "digital-scheduling-policy=sliding-window "
                    "digital-window-size=1 mesh-rows=1 mesh-cols=4 "
                    "arrays-per-core=1 array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        assignments = [
            assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
            if assignment["lane_kind"] == "digital"
        ]
        self.assertEqual([item["tile_id"] for item in assignments], [0, 1])
        self.assertEqual(
            report["plan"]["digital_scheduling_policy"], "sliding-window"
        )
        self.assertEqual(report["plan"]["digital_window_size"], 1)

    def test_sliding_window_preserves_parallel_work_within_window(self):
        planned = run_ir_pipeline(
            DIGITAL_FORK_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "digital-scheduling-policy=sliding-window "
                    "digital-window-size=2 mesh-rows=1 mesh-cols=4 "
                    "arrays-per-core=1 array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        assignments = [
            assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
            if assignment["lane_kind"] == "digital"
        ]
        self.assertEqual(len({item["tile_id"] for item in assignments}), 2)
        self.assertEqual(
            {item["start_ns"] for item in assignments},
            {assignments[0]["start_ns"]},
        )

    def test_first_use_window_binds_matrices_in_scheduled_order(self):
        case = LoweringCase(
            name="first_use_window_matrix_homes",
            model_factory=ThreeLayerRouteLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=8,
        )
        planned = run_pipeline(
            case,
            [
                *linear_prefix(array_cols=8),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=first-use-window "
                    "setup-binding-policy=consumer-anchored "
                    "digital-scheduling-policy=sliding-window "
                    "digital-window-size=1 mesh-rows=1 mesh-cols=4 "
                    "arrays-per-core=1 array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        self.assertEqual(
            report["plan"]["mvm_body_policy"], "first-use-window"
        )
        operations = {item["id"]: item for item in report["operations"]}
        assignments_by_operation: dict[int, set[int]] = {}
        for assignment in report["plan"]["realization"]["leaf_assignments"]:
            assignments_by_operation.setdefault(
                assignment["operation_id"], set()
            ).add(assignment["tile_id"])
        mvm_tiles = [
            next(iter(assignments_by_operation[operation_id]))
            for operation_id in sorted(operations)
            if operations[operation_id]["kind"] == "physical_mvm"
        ]
        self.assertEqual(mvm_tiles, [0, 1, 2])

        for group in report["lane_binding_groups"]:
            homes = {
                tile
                for operation_id in group["operation_ids"]
                for tile in assignments_by_operation.get(operation_id, set())
            }
            self.assertEqual(len(homes), 1)

    def test_first_use_window_requires_sliding_digital_policy(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mvm-body-policy=first-use-window "
                    "digital-scheduling-policy=earliest-finish "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8"
                ),
            ],
            input=DIGITAL_CHAIN_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "mvm-body-policy=first-use-window requires "
            "digital-scheduling-policy=sliding-window",
            result.stderr,
        )

    def test_first_use_adaptive_spills_to_preserve_ra_parallelism(self):
        case = LoweringCase(
            name="first_use_adaptive_parallel_branches",
            model_factory=ParallelLinearBranches,
            input_factory=lambda: (torch.ones(1, 8), torch.ones(1, 8)),
            array_rows=8,
            array_cols=8,
        )

        def plan(policy: str) -> dict:
            planned = run_pipeline(
                case,
                [
                    *linear_prefix(array_cols=8),
                    "--sculptor-build-ra-tree",
                    (
                        "--sculptor-plan-mapping="
                        "strategies=setup-first,recursive-fork-join "
                        f"mvm-body-policy={policy} "
                        "setup-binding-policy=consumer-anchored "
                        "digital-scheduling-policy=sliding-window "
                        "digital-window-size=1 mesh-rows=1 mesh-cols=2 "
                        "arrays-per-core=2 array-rows=8 array-cols=8 "
                        "verify-plan"
                    ),
                ],
            )
            return build_ra_tree_report(planned)["functions"][0]

        strict = plan("first-use-window")
        adaptive = plan("first-use-adaptive")

        def physical_mvm_assignments(report: dict) -> list[dict]:
            operation_kinds = {
                operation["id"]: operation["kind"]
                for operation in report["operations"]
            }
            return [
                assignment
                for assignment in report["plan"]["realization"][
                    "leaf_assignments"
                ]
                if operation_kinds[assignment["operation_id"]]
                == "physical_mvm"
            ]

        strict_mvms = physical_mvm_assignments(strict)
        adaptive_mvms = physical_mvm_assignments(adaptive)
        self.assertEqual(len(strict_mvms), 2)
        self.assertEqual(len(adaptive_mvms), 2)
        self.assertEqual(len({item["tile_id"] for item in strict_mvms}), 1)
        self.assertEqual(len({item["tile_id"] for item in adaptive_mvms}), 2)
        self.assertGreater(
            max(item["start_ns"] for item in strict_mvms),
            min(item["start_ns"] for item in strict_mvms),
        )
        self.assertEqual(
            {item["start_ns"] for item in adaptive_mvms},
            {adaptive_mvms[0]["start_ns"]},
        )
        self.assertLess(
            adaptive["plan"]["estimated_latency_ns"],
            strict["plan"]["estimated_latency_ns"],
        )
        self.assertEqual(
            adaptive["plan"]["mvm_body_policy"], "first-use-adaptive"
        )

    def test_first_use_adaptive_requires_sliding_digital_policy(self):
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mvm-body-policy=first-use-adaptive "
                    "digital-scheduling-policy=earliest-finish "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8"
                ),
            ],
            input=DIGITAL_CHAIN_FIXTURE,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "mvm-body-policy=first-use-adaptive requires "
            "digital-scheduling-policy=sliding-window",
            result.stderr,
        )

    def test_analog_row_views_bufferize_without_load_or_result_copies(self):
        case = LoweringCase(
            name="zero_copy_analog_row_views",
            model_factory=FullAnalogOccupancyWithDigitalTail,
            input_factory=lambda: (torch.ones(4, 8),),
            array_rows=8,
            array_cols=8,
        )
        lowered = run_serialized_pipeline(
            case,
            [
                *linear_prefix(array_cols=8),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=packed "
                    "setup-binding-policy=consumer-anchored "
                    "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=1 mesh-cols=1 "
                    "arrays-per-core=4 verify-placement"
                ),
                "--sculptor-outline-tile-routines",
            ],
            [
                "--sculptor-extract-tile-module=tile-id=0",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-finalize-tile-runtime-graph",
                "--sculptor-lower-golem-to-llvm-shims",
                "--canonicalize",
                "--cse",
                "--empty-tensor-to-alloc-tensor",
                (
                    "--one-shot-bufferize=bufferize-function-boundaries "
                    "function-boundary-type-conversion=identity-layout-map"
                ),
                "--buffer-results-to-out-params=hoist-static-allocs",
                "--convert-bufferization-to-memref",
                "--sculptor-bind-tile-routine-destinations",
                "--buffer-hoisting",
                "--buffer-loop-hoisting",
                "--buffer-deallocation-pipeline",
                "--optimize-allocation-liveness",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
        )
        lines = [line.strip() for line in lowered.splitlines()]
        load_calls = [
            index
            for index, line in enumerate(lines)
            if "func.call @golem_analog_mvm_load" in line
        ]
        self.assertGreater(len(load_calls), 0)
        for index in load_calls:
            self.assertGreaterEqual(index, 2)
            self.assertIn("memref.cast", lines[index - 1])
            self.assertIn("memref.subview %arg", lines[index - 2])
            self.assertNotIn(
                "memref.copy", "\n".join(lines[index - 2 : index])
            )
        self.assertIn(
            "sculptor.memory.eliminated_result_copies = 1 : i64", lowered
        )
        self.assertIn("planned_analog_store_copy_count = 1 : i64", lowered)
        self.assertIn("unplanned_full_tensor_copy_count = 0 : i64", lowered)

    def test_contiguous_linear_gather_canonicalizes_to_tensor_slice(self):
        lowered = run_ir_pipeline(
            CONTIGUOUS_GATHER_FIXTURE,
            ["--sculptor-canonicalize-layers"],
        )
        self.assertIn("tensor.extract_slice %arg0[4] [4] [1]", lowered)
        self.assertNotIn("tensor.extract %arg0", lowered)
        self.assertNotIn("linalg.index", lowered)

    def test_noncontiguous_gather_remains_explicit(self):
        fixture = CONTIGUOUS_GATHER_FIXTURE.replace(
            "%offset = arith.constant 4 : i64\n"
            "      %shifted = arith.addi %index_i64, %offset : i64",
            "%offset = arith.constant 2 : i64\n"
            "      %shifted = arith.muli %index_i64, %offset : i64",
        )
        lowered = run_ir_pipeline(
            fixture,
            ["--sculptor-canonicalize-layers"],
        )
        self.assertNotIn("tensor.extract_slice %arg0", lowered)
        self.assertIn("tensor.extract %arg0", lowered)

    def test_packed_bindings_share_partially_filled_tiles_across_waves(self):
        case = LoweringCase(
            name="packed_six_independent_waves",
            model_factory=SixLayerRouteLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=8,
        )
        planned = run_pipeline(
            case,
            [
                *linear_prefix(array_cols=8),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=packed "
                    "setup-binding-policy=consumer-anchored "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=4 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(planned)["functions"][0]
        physical_mvm_ids = {
            operation["id"]
            for operation in report["operations"]
            if operation["kind"] == "physical_mvm"
        }
        assignments = [
            assignment
            for assignment in report["plan"]["realization"][
                "leaf_assignments"
            ]
            if assignment["operation_id"] in physical_mvm_ids
        ]
        self.assertEqual(len(assignments), 6)
        self.assertEqual(len({item["tile_id"] for item in assignments}), 2)
        self.assertEqual(
            len({(item["tile_id"], item["lane_index"]) for item in assignments}),
            6,
        )

    def test_consumer_anchoring_keeps_cpu_usable_when_all_arrays_are_bound(self):
        case = LoweringCase(
            name="consumer_anchored_full_analog_occupancy",
            model_factory=FullAnalogOccupancyWithDigitalTail,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=2,
        )
        placed = run_pipeline(
            case,
            [
                *linear_prefix(array_cols=2),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=packed "
                    "setup-binding-policy=consumer-anchored "
                    "digital-scheduling-policy=progressive "
                    "mesh-rows=1 mesh-cols=1 "
                    "arrays-per-core=4 array-rows=8 array-cols=2 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=1 mesh-cols=1 "
                    "arrays-per-core=4 verify-placement"
                ),
            ],
        )
        self.assertIn(
            'sculptor.mapping.setup_binding_policy = "consumer-anchored"',
            placed,
        )
        self.assertIn("linalg.generic", placed)
        self.assertNotIn("consumer-unreserved core", placed)

    def test_shared_destination_initializer_is_not_mutated_by_routines(self):
        outlined = run_ir_pipeline(
            SHARED_DESTINATION_INITIALIZER_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=1 mesh-cols=2 arrays-per-core=4 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=1 mesh-cols=2 "
                    "arrays-per-core=4 verify-placement"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        tile_ids = sorted(
            {int(value) for value in re.findall(r"module @tile_(\d+)", outlined)}
        )
        lowered_tiles = [
            run_ir_pipeline(
                outlined,
                [
                    f"--sculptor-extract-tile-module=tile-id={tile_id}",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-finalize-tile-runtime-graph",
                "--empty-tensor-to-alloc-tensor",
                (
                    "--one-shot-bufferize=bufferize-function-boundaries "
                    "function-boundary-type-conversion=identity-layout-map"
                ),
                "--buffer-results-to-out-params=hoist-static-allocs",
                "--convert-bufferization-to-memref",
                "--sculptor-bind-tile-routine-destinations",
                ],
            )
            for tile_id in tile_ids
        ]
        lowered = "\n".join(lowered_tiles)

        matmul_routines = re.findall(
            r"func\.func private @routine_\d+\((.*?)\n  }",
            lowered,
            re.DOTALL,
        )
        matmul_routines = [
            routine for routine in matmul_routines if "linalg.matmul" in routine
        ]
        self.assertEqual(len(matmul_routines), 2)
        for routine in matmul_routines:
            seed = re.search(
                r"memref\.copy (%arg\d+), (%arg\d+) :", routine
            )
            output = re.search(
                r"linalg\.matmul .* outs\((%arg\d+)", routine
            )
            self.assertIsNotNone(seed)
            self.assertIsNotNone(output)
            self.assertNotEqual(seed.group(1), seed.group(2))
            self.assertEqual(output.group(1), seed.group(2))
            self.assertLess(seed.start(), output.start())
            self.assertEqual(routine.count("memref.copy"), 1)

    def test_placement_memory_estimate_tracks_lifetimes_and_fanout(self):
        prefix = [
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=recursive-fork-join "
                "mesh-rows=1 mesh-cols=1 arrays-per-core=1 "
                "array-rows=8 array-cols=8 verify-plan"
            ),
            (
                "--sculptor-place-logical-tiles="
                "schedule=greedy mesh-rows=1 mesh-cols=1 "
                "arrays-per-core=1 verify-placement"
            ),
        ]
        placed = run_ir_pipeline(LIFETIME_CHAIN_FIXTURE, prefix)
        estimate = re.search(
            r"#sculptor\.logical_tile_memory_estimate<"
            r"logicalTileId = 0 : i64, persistentBytes = (\d+) : i64, "
            r"producedBytes = (\d+) : i64, incomingBytes = (\d+) : i64, "
            r"requiredBytes = (\d+) : i64, complete = true>",
            placed,
        )
        self.assertIsNotNone(estimate)
        persistent, produced, incoming, required = map(int, estimate.groups())
        # Three 64-byte intermediates are touched, but only two overlap.
        self.assertEqual((persistent, produced, incoming, required),
                         (0, 128, 0, 128))
        self.assertIn(
            'sculptor.mapping.memory_estimate_method = '
            '"schedule-aware-lifetimes-v1"',
            placed,
        )

        fanout = run_ir_pipeline(
            LIFETIME_FANOUT_FIXTURE,
            [
                prefix[0],
                (
                    "--sculptor-plan-mapping="
                    "strategies=consumer-bound-fill "
                    "mesh-rows=1 mesh-cols=1 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                prefix[2],
            ],
        )
        fanout_required = re.search(
            r"logicalTileId = 0 : i64, persistentBytes = 0 : i64, "
            r"producedBytes = (\d+) : i64, incomingBytes = 0 : i64, "
            r"requiredBytes = (\d+) : i64, complete = true>",
            fanout,
        )
        self.assertIsNotNone(fanout_required)
        # The shared value remains live through both branches.
        self.assertEqual(tuple(map(int, fanout_required.groups())), (192, 192))

        outlined = run_ir_pipeline(
            LIFETIME_CHAIN_FIXTURE,
            [*prefix, "--sculptor-outline-tile-routines"],
        )
        exact = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-extract-tile-module=tile-id=0",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-plan-tile-scratchpad=bytes=128 alignment=16",
            ],
            input=outlined,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(exact.returncode, 0, exact.stderr)
        exact_required = re.search(
            r"#sculptor\.tile_memory_capacity<[^>]*"
            r"requiredLocalBytes = (\d+) : i64",
            exact.stdout,
        )
        self.assertIsNotNone(exact_required)
        self.assertLessEqual(int(exact_required.group(1)), required)

    def test_concat_dependencies_charge_each_producer_shard_once(self):
        planned = run_ir_pipeline(
            CONCAT_PRODUCER_CONTRIBUTION_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=1 mesh-cols=3 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        dependencies = re.findall(
            r"#sculptor\.logical_tile_dependency<"
            r"sourceOperationId = ([01]) : i64, sourceWorkUnitId = -1 : i64, "
            r"targetOperationId = 2 : i64, targetWorkUnitId = -1 : i64, "
            r"tensorId = (\d+) : i64, targetOperandNumber = -1 : i64, "
            r"byteSize = (\d+) : i64>",
            planned,
        )
        self.assertEqual(len(dependencies), 2)
        self.assertEqual({int(source) for source, _, _ in dependencies}, {0, 1})
        self.assertEqual({int(size) for _, _, size in dependencies}, {32})
        self.assertEqual(sum(int(size) for _, _, size in dependencies), 64)

    def test_minimum_work_grain_avoids_small_operation_oversharding(self):
        case = LoweringCase(
            name="adaptive_digital_grain",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        fine = run_pipeline(
            case,
            ["--sculptor-expand-digital-work=parallel-workers=4"],
        )
        coarse = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 minimum-work-items-per-unit=64"
            ],
        )
        self.assertIn(
            "sculptor.mapping.expanded_digital_work_unit_count = 8", fine
        )
        self.assertIn(
            "sculptor.mapping.expanded_digital_work_unit_count = 0", coarse
        )
        self.assertIn(
            "sculptor.mapping.digital_minimum_work_items_per_unit = 64",
            coarse,
        )

    def test_inline_splat_matrix_stays_compact_when_tiled(self):
        lowered = run_ir_pipeline(
            INLINE_SPLAT_MVM_FIXTURE,
            ["--sculptor-expand-mvm-to-golem=array-rows=8 array-cols=2"],
        )
        self.assertNotIn("sculptor.mvm ", lowered)
        self.assertEqual(lowered.count("sculptor.array.set "), 2)
        self.assertEqual(lowered.count("sculptor.array.execute "), 2)
        self.assertIn('sculptor.source_resource = "inline_dense_matrix_0"', lowered)
        self.assertEqual(lowered.count("dense<1.000000e+00>"), 2)
        self.assertEqual(lowered.count("tensor<4x2xf32>"), 4)
        self.assertNotIn("dense_resource<inline_dense_matrix_0__tile_", lowered)

    def test_static_tensor_pad_has_a_static_mapping_iteration_domain(self):
        planned = run_ir_pipeline(
            STATIC_PAD_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
            ],
        )
        self.assertIn("#sculptor.mapping_plan<", planned)
        self.assertIn("tensor.pad", planned)

    def test_static_tensor_pad_region_is_owned_by_each_outlined_work_unit(self):
        outlined = run_ir_pipeline(
            STATIC_PAD_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=2 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=2 mesh-cols=2 "
                    "arrays-per-core=1"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("module @tile_", outlined)
        self.assertEqual(outlined.count("tensor.pad"), 2)
        self.assertEqual(outlined.count("tensor.generate"), 2)
        self.assertEqual(outlined.count("tensor.yield"), 4)

    def test_static_view_slice_is_routed_before_tiled_tensor_pad(self):
        outlined = run_ir_pipeline(
            STATIC_VIEW_CHAIN_PAD_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=2 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=1"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        pad_routines = re.findall(
            r"func\.func private @routine_\d+\("
            r"%arg0: tensor<1x2x4x4xf32>\)"
            r" -> tensor<1x2x6x6xf32>[\s\S]*?"
            r"tensor\.pad %arg0",
            outlined,
        )
        self.assertEqual(len(pad_routines), 2)
        self.assertNotRegex(
            outlined,
            r"func\.func private @routine_\d+\("
            r"%arg0: tensor<1x8x4x4xf32>\)"
            r" -> tensor<1x2x6x6xf32>",
        )

    def setUp(self):
        self.linear_case = LoweringCase(
            name="mapping_extension_linear",
            model_factory=FourWayReductionLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=2,
        )

    def test_mvm_wave_planner_builds_a_verified_temporal_schedule(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,mvm-wave "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 verify-plan"
                ),
            ],
        )
        self.assertIn('planner = "setup-first,mvm-wave"', lowered)
        self.assertIn("kind = temporal_cut", lowered)

    def test_setup_first_is_mandatory_and_survives_recursive_fork_join(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 verify-plan"
                ),
            ],
        )
        report = build_ra_tree_report(lowered)["functions"][0]
        self.assertEqual(
            report["plan"]["planner"],
            "setup-first,recursive-fork-join",
        )

        operations = {
            operation["id"]: operation for operation in report["operations"]
        }
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        root = nodes[report["tree"]["root_id"]]
        self.assertEqual(root["kind"], "temporal_cut")
        self.assertEqual(len(root["child_ids"]), 2)

        def leaf_operations(node_id: int) -> list[int]:
            node = nodes[node_id]
            if node["operation_id"] >= 0:
                return [node["operation_id"]]
            return [
                operation_id
                for child_id in node["child_ids"]
                for operation_id in leaf_operations(child_id)
            ]

        setup_ids = leaf_operations(root["child_ids"][0])
        compute_ids = leaf_operations(root["child_ids"][1])
        self.assertGreater(len(setup_ids), 1)
        self.assertEqual(nodes[root["child_ids"][0]]["kind"], "spatial_cut")
        self.assertTrue(
            all(operations[operation_id]["kind"] == "matrix_setup"
                for operation_id in setup_ids)
        )
        self.assertTrue(
            all(operations[operation_id]["kind"] != "matrix_setup"
                for operation_id in compute_ids)
        )
        self.assertEqual(
            set(setup_ids),
            {
                operation_id
                for operation_id, operation in operations.items()
                if operation["kind"] == "matrix_setup"
            },
        )

    def test_temporal_verifier_rejects_a_reversed_dependency_chain(self):
        built = run_ir_pipeline(
            LIFETIME_CHAIN_FIXTURE,
            ["--sculptor-build-ra-tree"],
        )
        reversed_tree = built.replace(
            "childIds = [0, 1, 2, 3]",
            "childIds = [3, 2, 1, 0]",
            1,
        )
        self.assertNotEqual(reversed_tree, built)
        with self.assertRaisesRegex(
            AssertionError,
            r"orders mapping dependency 0:-1 -> 1:-1 backwards",
        ):
            run_ir_pipeline(
                reversed_tree,
                [
                    (
                        "--sculptor-plan-mapping="
                        "strategies=recursive-fork-join "
                        "mesh-rows=2 mesh-cols=2 arrays-per-core=4 "
                        "array-rows=8 array-cols=8"
                    )
                ],
            )

    def test_calibrated_profile_and_makespan_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "placement.csv"
            lowered = run_pipeline(
                self.linear_case,
                [
                    *linear_prefix(),
                    "--sculptor-build-ra-tree",
                    (
                        "--sculptor-plan-mapping="
                        "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                        "arrays-per-core=4 array-rows=8 array-cols=2 "
                        f"cost-profile={COST_PROFILE}"
                    ),
                    (
                        "--sculptor-place-logical-tiles="
                        "schedule=greedy objective=makespan network-mode=full "
                        "timing-scope=warm temporal-candidate-limit=4 "
                        "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                        f"summary-output={summary}"
                    ),
                ],
            )
            summary_text = summary.read_text()

        expected_hash = hashlib.sha256(COST_PROFILE.read_bytes()).hexdigest()
        self.assertIn('costProfileName = "test-calibrated-v1"', lowered)
        self.assertIn(f'costProfileHash = "{expected_hash}"', lowered)
        self.assertIn("cost_profile_name,cost_profile_hash", summary_text)
        self.assertIn(f"test-calibrated-v1,{expected_hash}", summary_text)
        self.assertIn('objective = "makespan"', lowered)
        self.assertIn('networkMode = "full"', lowered)
        self.assertIn('timingScope = "warm"', lowered)
        makespan = re.search(r"predictedMakespanNs = ([0-9.eE+-]+)", lowered)
        self.assertIsNotNone(makespan)
        self.assertGreater(float(makespan.group(1)), 0.0)

    def test_incremental_makespan_annealing_matches_full_evaluator(self):
        prefix = [
            *linear_prefix(),
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=setup-first,recursive-fork-join "
                "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                "array-rows=8 array-cols=2 "
                f"cost-profile={COST_PROFILE}"
            ),
        ]
        placement = (
            "schedule=annealing annealing-initial-schedule=greedy "
            "annealing-iterations=200 annealing-initial-temperature=0 "
            "annealing-cooling-rate=0.995 "
            "annealing-trace-sample-interval=20 random-seed=7 "
            "objective=makespan network-mode=full timing-scope=warm "
            "temporal-candidate-limit=4 "
            "greedy-tile-order=priority greedy-priority-mode=max "
            "greedy-candidate-scope=frontier greedy-lookahead=2 "
            "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
        )
        full = run_pipeline(
            self.linear_case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    f"{placement} annealing-incremental-makespan=false"
                ),
            ],
        )
        incremental = run_pipeline(
            self.linear_case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    f"{placement} annealing-incremental-makespan=true "
                    "annealing-makespan-verify-interval=1"
                ),
            ],
        )

        self.assertEqual(incremental, full)
        self.assertIn("evaluations = 200 : i64", incremental)
        self.assertIn("logical_tile_annealing_trace", incremental)

    def test_default_extension_modes_remain_legacy(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-expand-digital-work=parallel-workers=2",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )

        self.assertIn('costProfileName = "legacy-v1"', lowered)
        self.assertIn('objective = "transfer-cost"', lowered)
        self.assertIn('dataflow_mode = "bulk"', lowered)
        self.assertIn('reduction_tree_policy = "none"', lowered)
        self.assertRegex(lowered, r"predictedMakespanNs = 0(?:\.0+)?")

    def test_temporal_placement_survives_serialization_before_outlining(self):
        for index, (network_mode, timing_scope) in enumerate(
            (network, scope)
            for network in ("ideal", "finite", "full")
            for scope in ("warm", "cold")
        ):
            with self.subTest(
                dataflow="bulk",
                network_mode=network_mode,
                timing_scope=timing_scope,
            ):
                outlined = run_serialized_pipeline(
                    self.linear_case,
                    [
                        *linear_prefix(),
                        "--sculptor-build-ra-tree",
                        (
                            "--sculptor-plan-mapping="
                            "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                            "arrays-per-core=4 array-rows=8 array-cols=2 "
                            f"cost-profile={COST_PROFILE}"
                        ),
                        (
                            "--sculptor-place-logical-tiles="
                            "schedule=greedy objective=makespan "
                            f"network-mode={network_mode} "
                            f"timing-scope={timing_scope} "
                            f"temporal-candidate-limit={1 + index} "
                            "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                        ),
                    ],
                    ["--sculptor-outline-tile-routines"],
                )
                self.assertIn("module @tile_", outlined)

        sharded_case = LoweringCase(
            name="serialized_sharded_elementwise_chain",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        outlined = run_serialized_pipeline(
            sharded_case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 "
                    f"cost-profile={COST_PROFILE}"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy objective=makespan network-mode=full "
                    "timing-scope=warm temporal-candidate-limit=4 "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                ),
            ],
            ["--sculptor-outline-tile-routines"],
        )
        self.assertIn("module @tile_", outlined)
        self.assertIn("sculptor.memory.plan_version = 3 : i64", outlined)
        self.assertIn("#sculptor.tile_memory_owner<", outlined)
        self.assertIn("#sculptor.tile_memory_view<", outlined)

        outlined = run_serialized_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 dataflow=sharded "
                    "reduction-tree=balanced reduction-fan-in=2 "
                    "reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2 "
                    f"cost-profile={COST_PROFILE}"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy objective=makespan network-mode=finite "
                    "timing-scope=cold temporal-candidate-limit=4 "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4"
                ),
            ],
            ["--sculptor-outline-tile-routines"],
        )
        self.assertIn("sculptor.task.reduction_tree_id", outlined)

    def test_tile_memory_capacity_is_serialized_and_enforced(self):
        prefix = [
            *linear_prefix(),
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                "arrays-per-core=4 array-rows=8 array-cols=2"
            ),
        ]
        unconstrained = run_pipeline(
            self.linear_case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        estimates = [
            int(value)
            for value in re.findall(
                r"requiredBytes = ([0-9]+) : i64", unconstrained
            )
        ]
        self.assertTrue(estimates)
        capacity = max(estimates)
        placement = (
            "--sculptor-place-logical-tiles="
            "schedule=greedy mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
            f"tile-memory-capacity-bytes={capacity}"
        )
        first = run_pipeline(self.linear_case, [*prefix, placement])
        second = run_pipeline(self.linear_case, [*prefix, placement])
        self.assertEqual(first, second)
        self.assertIn(f"tileMemoryCapacityBytes = {capacity} : i64", first)
        self.assertRegex(
            first,
            r"logicalTileId = 0 : i64, persistentBytes = [1-9][0-9]* : i64",
        )
        outlined = run_serialized_pipeline(
            self.linear_case,
            [*prefix, placement],
            ["--sculptor-outline-tile-routines"],
        )
        self.assertIn("module @tile_", outlined)

        failing_placement = (
            "--sculptor-place-logical-tiles="
            "schedule=greedy mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
            f"tile-memory-capacity-bytes={capacity - 1}"
        )
        result = subprocess.run(
            [str(DEFAULT_SCULPTOR_OPT), "-", *prefix, failing_placement],
            input=export_linalg(self.linear_case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("logical tile 0 requires", result.stderr)
        self.assertIn(f"capacity of {capacity - 1} bytes", result.stderr)

    def test_capacity_accounts_for_returned_value_reused_internally(self):
        placed = run_pipeline(
            RETURNED_AND_REUSED_OUTPUT_FIXTURE,
            [
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=4 minimum-work-items-per-unit=1 "
                    "dataflow=sharded"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=8"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 tile-memory-capacity-bytes=1048576"
                ),
            ],
        )
        self.assertIn("complete = true", placed)
        self.assertRegex(placed, r"producedBytes = [1-9][0-9]* : i64")

    def test_capacity_aware_output_placement_and_exact_validation(self):
        case = LoweringCase(
            name="capacity_aware_output",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        prefix = [
            "--sculptor-expand-digital-work="
            "parallel-workers=4 dataflow=sharded",
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=recursive-fork-join "
                "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                "array-rows=8 array-cols=2"
            ),
        ]
        outlined = run_pipeline(
            case,
            [
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 tile-memory-capacity-bytes=512"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertRegex(
            outlined,
            r"sculptor\.deployment\.model_outputs = "
            r"\[#sculptor\.tile_routine_model_io<[^>]*tile = 5 : i64",
        )
        required = [
            int(value)
            for value in re.findall(
                r"#sculptor\.tile_memory_capacity<[^>]*"
                r"requiredLocalBytes = ([0-9]+) : i64",
                outlined,
            )
        ]
        self.assertTrue(required)
        self.assertEqual(max(required), 320)

        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                *prefix,
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 tile-memory-capacity-bytes=287"
                ),
                "--sculptor-outline-tile-routines",
                "--sculptor-extract-tile-module=tile-id=0",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-finalize-tile-runtime-graph",
            ],
            input=export_linalg(case),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "physical tile 0 requires 288 finalized local-memory bytes",
            result.stderr,
        )
        self.assertIn(
            "exceeding the configured capacity of 287 bytes", result.stderr
        )

    def test_temporal_placement_reports_specific_metadata_mismatch(self):
        placed = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 array-rows=8 array-cols=2 "
                    f"cost-profile={COST_PROFILE}"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy objective=makespan network-mode=full "
                    "timing-scope=warm mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        corrupted = placed.replace(
            'costProfileName = "test-calibrated-v1"',
            'costProfileName = "incorrect-profile"',
            1,
        )
        self.assertNotEqual(corrupted, placed)
        result = subprocess.run(
            [str(DEFAULT_SCULPTOR_OPT), "-", "--sculptor-outline-tile-routines"],
            input=corrupted,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "logical-tile placement cost profile name mismatch", result.stderr
        )

    def test_sharded_elementwise_chain_has_exact_edges(self):
        case = LoweringCase(
            name="sharded_elementwise_chain",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        lowered = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
            ],
        )

        self.assertIn('dataflow_mode = "sharded"', lowered)
        self.assertRegex(lowered, r"shardGroupId = [0-9]+")
        self.assertRegex(lowered, r"shardCount = 4")
        self.assertIn("sculptor.mapping.shard_group_count = 1", lowered)
        self.assertIn("sculptor.mapping.shard_edge_count = 4", lowered)
        self.assertIn("sculptor.mapping.assembly_boundary_count = 1", lowered)
        self.assertIn("#sculptor.mapping_work_unit_edge<", lowered)
        self.assertRegex(lowered, r"targetOperandNumber = [0-9]+")
        self.assertNotIn("tensorId = -1", lowered)

        outlined = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("byteSize = 32 : i64", outlined)
        self.assertIn("sculptor.memory.plan_version = 3 : i64", outlined)
        self.assertIn("#sculptor.tile_memory_owner<", outlined)
        self.assertNotIn("kind = local_temporary", outlined)
        self.assertIn("#sculptor.tile_memory_view<", outlined)
        self.assertIn("effect = read", outlined)
        self.assertIn("effect = write", outlined)
        self.assertIn("mode = local_alias", outlined)
        self.assertIn("mode = contiguous", outlined)
        self.assertNotIn("mode = packed", outlined)
        self.assertNotIn("mode = segmented", outlined)
        self.assertRegex(
            outlined,
            r"#sculptor\.tile_memory_assembly<[^>]*"
            r"contributingViewIds = \[[0-9]+, [0-9]+, [0-9]+, [0-9]+\], "
            r"destinationViewIds = \[[0-9]+, [0-9]+, [0-9]+, [0-9]+\], "
            r"completionEventIds = \[[0-9]+, [0-9]+, [0-9]+, [0-9]+\], "
            r"readinessEventId = [0-9]+ : i64, "
            r"localCopyBytes = 32 : i64, routedBytes = 96 : i64>",
        )
        self.assertIn("kind = assembly_ready", outlined)
        self.assertIn("sculptor.memory.event_edges", outlined)
        self.assertIn("sculptor.memory.lifetimes", outlined)
        self.assertIn("sculptor.memory.interferences", outlined)
        self.assertIn("relation = interferes", outlined)
        self.assertRegex(
            outlined,
            r"#sculptor\.tile_memory_capacity<[^>]*"
            r"routeInputBytes = [0-9]+ : i64, "
            r"routeOutputBytes = [0-9]+ : i64, "
            r"assemblyBytes = [0-9]+ : i64, "
            r"intermediateBytes = [0-9]+ : i64, "
            r"routineTemporaryPeakBytes = [0-9]+ : i64, "
            r"routineTemporaryTotalBytes = [0-9]+ : i64, "
            r"peakLiveBytes = [0-9]+ : i64, "
            r"requiredLocalBytes = [0-9]+ : i64, "
            r"reusableBytes = [0-9]+ : i64, complete = true>",
        )
        for edge_kind in (
            "routine_execution",
            "local_dependency",
            "route_send",
            "network_transfer",
            "route_ready",
            "assembly_contribution",
            "dma_completion",
            "assembly_join",
        ):
            self.assertIn(f"kind = {edge_kind}", outlined)
        self.assertEqual(outlined.count("sculptor.memory.zero_copy_view"), 8)
        self.assertRegex(
            outlined,
            r"tensor\.extract_slice[^\n]*"
            r"sculptor\.memory\.owner_id = [0-9]+ : i64[^\n]*"
            r"sculptor\.memory\.view_id = [0-9]+ : i64[^\n]*"
            r"sculptor\.memory\.zero_copy_view",
        )
        materialized = materialize_outlined_tiles(outlined)
        self.assertTrue(
            any("task_graph.route_input" in ir for ir in materialized)
        )
        self.assertTrue(
            any("task_graph.route_output" in ir for ir in materialized)
        )
        self.assertTrue(
            any("sculptor.memory.assemblies" in ir for ir in materialized)
        )
        tile_abi = emit_outlined_tile_abi(outlined, 0)
        for symbol in (
            "@__golem_tile_memory_owners",
            "@__golem_tile_memory_views",
            "@__golem_tile_memory_view_geometry",
            "@__golem_tile_route_views",
            "@__golem_tile_assemblies",
            "@__golem_tile_assembly_contributions",
            "@golem_tile_memory_owner_count",
            "@golem_tile_route_view_count",
            "@golem_tile_assembly_count",
        ):
            self.assertIn(symbol, tile_abi)
        self.assertNotIn("#sculptor.", tile_abi)
        self.assertNotIn('"sculptor.', tile_abi)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "outlined.mlir"
            source.write_text(outlined)
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    str(source),
                    "--sculptor-extract-tile-module=tile-id=0",
                    "--sculptor-materialize-tile-runtime-graph",
                    "--sculptor-finalize-tile-runtime-graph",
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=bufferize-function-boundaries "
                    "function-boundary-type-conversion=identity-layout-map",
                    "--buffer-results-to-out-params=hoist-static-allocs",
                    "--convert-bufferization-to-memref",
                    "--sculptor-bind-tile-routine-destinations",
                    "--buffer-hoisting",
                    "--buffer-loop-hoisting",
                    "--buffer-deallocation-pipeline",
                    "--optimize-allocation-liveness",
                    "--sculptor-audit-tile-bufferization=strict=true",
                    "--sculptor-vectorize-tile-copies=vector-bits=256",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        routine_zero = result.stdout.split(
            "func.func private @routine_0", 1
        )[1].split("func.func private @routine_4", 1)[0]
        self.assertEqual(routine_zero.count("memref.subview"), 2)
        self.assertNotIn("memref.alloc", routine_zero)
        self.assertNotIn("memref.copy", routine_zero)
        self.assertIn("outs(%arg2", routine_zero)
        self.assertIn("sculptor.memory.destination_bound", routine_zero)
        self.assertIn("sculptor.memory.destination_bindings", routine_zero)
        self.assertIn("sculptor.memory.owner_id", routine_zero)
        self.assertIn("sculptor.memory.view_id", routine_zero)

        routine_four = result.stdout.split(
            "func.func private @routine_4", 1
        )[1].split("func.func private @generate_task_graph", 1)[0]
        self.assertEqual(routine_four.count("memref.copy"), 0)
        self.assertEqual(routine_four.count("memref.subview"), 1)
        self.assertEqual(routine_four.count("vector.load"), 1)
        self.assertEqual(routine_four.count("vector.store"), 1)
        self.assertIn(
            "sculptor.memory.eliminated_result_copies = 3 : i64",
            result.stdout,
        )
        self.assertIn("sculptor.memory.bufferization_audit", result.stdout)
        self.assertIn("planned_assembly_copy_count = 1 : i64", result.stdout)
        self.assertIn("unplanned_full_tensor_copy_count = 0 : i64", result.stdout)
        self.assertIn("assembly_pack_count = 1 : i64", result.stdout)
        self.assertIn("vectorized_copy_count = 1 : i64", result.stdout)
        self.assertIn("vectorized_copy_bytes = 32 : i64", result.stdout)
        self.assertIn("externalBytes = 384 : i64", result.stdout)
        self.assertIn("workspaceBytes = 128 : i64", result.stdout)
        self.assertIn("assemblyBytes = 128 : i64", result.stdout)
        self.assertIn("routineTemporaryPeakBytes = 0 : i64", result.stdout)
        self.assertIn("allocation_count = 0 : i64", result.stdout)
        self.assertIn("peakLiveBytes = 512 : i64", result.stdout)
        self.assertIn("requiredLocalBytes = 512 : i64", result.stdout)
        assembly_owner = re.search(
            r"#sculptor\.tile_memory_assembly<[^>]*ownerId = ([0-9]+) : i64",
            result.stdout,
        )
        self.assertIsNotNone(assembly_owner)
        self.assertRegex(
            result.stdout,
            rf"#sculptor\.tile_memory_lifetime<[^>]*"
            rf"subjectKind = owner, storage = external, "
            rf"ownerId = {assembly_owner.group(1)} : i64[^>]*"
            rf"offset = -1 : i64",
        )
        route_input_owner = re.search(
            r"#sculptor\.tile_memory_owner<id = ([0-9]+) : i64[^>]*"
            r"kind = route_input",
            result.stdout,
        )
        self.assertIsNotNone(route_input_owner)
        self.assertRegex(
            result.stdout,
            rf"#sculptor\.tile_memory_lifetime<[^>]*"
            rf"subjectKind = owner, storage = workspace, "
            rf"ownerId = {route_input_owner.group(1)} : i64[^>]*"
            rf"offset = [0-9]+ : i64",
        )

        tile_marker = "module @tile_0 attributes"
        prefix, tile_zero = outlined.split(tile_marker, 1)
        corrupted = prefix + tile_marker + tile_zero.replace(
            "byteOffset = 0 : i64", "byteOffset = 4096 : i64", 1
        )
        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-extract-tile-module=tile-id=0",
            ],
            input=corrupted,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(
            result.stderr,
            r"memory view [0-9]+ lies outside owner [0-9]+ byte range",
        )

    def test_happens_before_reuses_only_proven_nonoverlapping_storage(self):
        case = LoweringCase(
            name="happens_before_workspace_reuse",
            model_factory=ForkedElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        producing_passes = [
            "--sculptor-expand-digital-work=parallel-workers=1",
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                "array-rows=8 array-cols=2"
            ),
            (
                "--sculptor-place-logical-tiles="
                "schedule=greedy mesh-rows=1 mesh-cols=1 "
                "arrays-per-core=4"
            ),
            "--sculptor-outline-tile-routines",
        ]
        consuming_passes = [
            "--sculptor-extract-tile-module=tile-id=0",
            "--sculptor-materialize-tile-runtime-graph",
            "--sculptor-finalize-tile-runtime-graph",
        ]
        first = run_serialized_pipeline(
            case, producing_passes, consuming_passes
        )
        second = run_serialized_pipeline(
            case, producing_passes, consuming_passes
        )
        self.assertEqual(first, second)

        workspace_lifetimes = {
            int(lifetime): int(offset)
            for lifetime, offset in re.findall(
                r"#sculptor\.tile_memory_lifetime<"
                r"id = (\d+) : i64, subjectKind = owner, "
                r"storage = workspace,.*?offset = (-?\d+) : i64",
                first,
                re.DOTALL,
            )
        }
        self.assertGreaterEqual(len(workspace_lifetimes), 4)
        self.assertTrue(
            all(offset >= 0 for offset in workspace_lifetimes.values())
        )

        relations = {
            (int(left), int(right)): relation
            for left, right, relation in re.findall(
                r"#sculptor\.tile_memory_interference<"
                r"id = \d+ : i64, leftLifetimeId = (\d+) : i64, "
                r"rightLifetimeId = (\d+) : i64, relation = (\w+)>",
                first,
            )
        }

        # Check the compiler's indexed DAG reachability against a deliberately
        # simple DFS oracle.  The extracted tile keeps deployment-wide event
        # IDs, so this also covers sparse event IDs in tile-local plans.
        successors: dict[int, list[int]] = {}
        for source, target in re.findall(
            r"#sculptor\.tile_memory_event_edge<id = \d+ : i64, "
            r"sourceEventId = (\d+) : i64, targetEventId = (\d+) : i64,",
            first,
        ):
            successors.setdefault(int(source), []).append(int(target))

        reachability_cache: dict[int, set[int]] = {}

        def reachable_from(source: int) -> set[int]:
            if source in reachability_cache:
                return reachability_cache[source]
            visited: set[int] = set()
            worklist = list(successors.get(source, ()))
            while worklist:
                event = worklist.pop()
                if event in visited:
                    continue
                visited.add(event)
                worklist.extend(successors.get(event, ()))
            reachability_cache[source] = visited
            return visited

        lifetime_accesses: dict[int, list[int]] = {}
        for record in re.findall(
            r"#sculptor\.tile_memory_lifetime<([^>]*)>", first
        ):
            lifetime = re.search(r"\bid = (\d+) : i64", record)
            access_events = re.search(
                r"\baccessEventIds = \[([^]]*)\]", record
            )
            self.assertIsNotNone(lifetime)
            self.assertIsNotNone(access_events)
            lifetime_accesses[int(lifetime.group(1))] = [
                int(event)
                for event in re.findall(r"-?\d+", access_events.group(1))
            ]

        def all_before(left: int, right: int) -> bool:
            left_accesses = lifetime_accesses[left]
            right_accesses = lifetime_accesses[right]
            return bool(left_accesses and right_accesses) and all(
                target in reachable_from(source)
                for source in left_accesses
                for target in right_accesses
            )

        for (left, right), relation in relations.items():
            if relation not in {"before", "after", "interferes"}:
                continue
            expected = (
                "before"
                if all_before(left, right)
                else "after"
                if all_before(right, left)
                else "interferes"
            )
            self.assertEqual(relation, expected)

        ordered_reuse = False
        concurrent_separation = False
        for (left, right), relation in relations.items():
            if (
                left not in workspace_lifetimes
                or right not in workspace_lifetimes
            ):
                continue
            same_offset = (
                workspace_lifetimes[left] == workspace_lifetimes[right]
            )
            if (
                relation in {"before", "after", "in_place_alias"}
                and same_offset
            ):
                ordered_reuse = True
            if relation == "interferes" and not same_offset:
                concurrent_separation = True
        self.assertTrue(ordered_reuse)
        self.assertTrue(concurrent_separation)

        capacity_match = re.search(
            r"#sculptor\.tile_memory_capacity<[^>]*"
            r"workspaceBytes = (\d+) : i64,.*?"
            r"intermediateBytes = (\d+) : i64,.*?"
            r"reusableBytes = (\d+) : i64, complete = true>",
            first,
        )
        self.assertIsNotNone(capacity_match)
        workspace_bytes, intermediate_bytes, reusable_bytes = map(
            int, capacity_match.groups()
        )
        self.assertLess(workspace_bytes, intermediate_bytes)
        self.assertEqual(
            intermediate_bytes - workspace_bytes, reusable_bytes
        )

    def test_route_final_use_events_hold_fan_out_and_consumers(self):
        fan_out_case = LoweringCase(
            name="route_final_fan_out",
            model_factory=TwoLayerFanOutLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=2,
        )
        outlined = run_pipeline(
            fan_out_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=spread mesh-rows=8 mesh-cols=8 "
                    "arrays-per-core=4 array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=snake mesh-rows=8 mesh-cols=8 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        routes = {
            tuple(map(int, match))
            for match in re.findall(
                r"#sculptor\.tile_routine_route<id = (\d+) : i64, "
                r"sourceTile = (\d+) : i64, sourceRoutine = (\d+) : i64, "
                r"sourceOutput = (\d+) : i64, destinationTile = (\d+) : i64, "
                r"destinationRoutine = \d+ : i64, destinationInput = \d+ : i64, "
                r"resourceId = (\d+) : i64, tensorId = (-?\d+) : i64",
                outlined,
            )
        }
        grouped_routes: dict[
            tuple[int, int, int], list[tuple[int, int, int, int]]
        ] = {}
        for (
            route_id,
            tile,
            routine,
            output,
            destination,
            resource,
            tensor,
        ) in routes:
            grouped_routes.setdefault((tile, routine, output), []).append(
                (route_id, destination, resource, tensor)
            )
        source, fan_out_routes = max(
            grouped_routes.items(), key=lambda item: len(item[1])
        )
        self.assertEqual(len(fan_out_routes), 4)
        fan_out_resources = {route[2] for route in fan_out_routes}
        self.assertEqual(len(fan_out_resources), 1)
        fan_out_tensor_ids = {
            route[3] for route in fan_out_routes if route[3] >= 0
        }
        self.assertTrue(fan_out_tensor_ids)
        expected_owner_tensor_id = (
            next(iter(fan_out_tensor_ids))
            if len(fan_out_tensor_ids) == 1
            else -1
        )
        owner = re.search(
            r"#sculptor\.tile_memory_owner<id = \d+ : i64, "
            rf"resourceId = {next(iter(fan_out_resources))} : i64, "
            rf"tensorId = (-?\d+) : i64, tile = {source[0]} : i64,",
            outlined,
        )
        self.assertIsNotNone(owner)
        self.assertEqual(int(owner.group(1)), expected_owner_tensor_id)

        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                f"--sculptor-extract-tile-module=tile-id={source[0]}",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-plan-tile-scratchpad=bytes=32 alignment=16",
                "--sculptor-finalize-tile-runtime-graph",
            ],
            input=outlined,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        completion_records = {
            tuple(match)
            for match in re.findall(
                r"#sculptor\.tile_memory_completion<id = (\d+) : i64, "
                r"kind = (\w+), tile = -?\d+ : i64, routine = -?\d+ : i64, "
                r"routeId = (-?\d+) : i64, ownerId = (-?\d+) : i64",
                result.stdout,
            )
        }
        send_records = [
            record for record in completion_records
            if record[1] == "route_send_complete"
            and int(record[2]) in {route[0] for route in fan_out_routes}
        ]
        self.assertEqual(len(send_records), 4)
        source_owner = {record[3] for record in send_records}
        self.assertEqual(len(source_owner), 1)
        fan_out_join = [
            record for record in completion_records
            if record[1] == "final_fan_out_send"
            and record[3] in source_owner
        ]
        scratchpad_release = [
            record for record in completion_records
            if record[1] == "scratchpad_release"
            and record[3] in source_owner
        ]
        dma_complete = [
            record for record in completion_records
            if record[1] == "dma_complete" and record[3] in source_owner
        ]
        self.assertEqual(len(fan_out_join), 1)
        self.assertEqual(len(scratchpad_release), 1)
        self.assertEqual(len(dma_complete), 1)
        event_edges = {
            (int(source_id), int(target_id), kind)
            for source_id, target_id, kind in re.findall(
                r"#sculptor\.tile_memory_event_edge<id = \d+ : i64, "
                r"sourceEventId = (\d+) : i64, targetEventId = (\d+) : i64, "
                r"kind = (\w+)>",
                result.stdout,
            )
        }
        join_id = int(fan_out_join[0][0])
        dma_id = int(dma_complete[0][0])
        release_id = int(scratchpad_release[0][0])
        for send in send_records:
            send_id = int(send[0])
            self.assertIn((send_id, join_id, "fan_out_join"), event_edges)
            self.assertIn((send_id, dma_id, "dma_completion"), event_edges)
        self.assertIn((dma_id, release_id, "scratchpad_release"), event_edges)

        tile_abi = emit_outlined_tile_abi(
            outlined,
            source[0],
            ("--sculptor-plan-tile-scratchpad=bytes=32 alignment=16",),
            strict_memory_audit=False,
        )
        self.assertIn("@__golem_tile_dma_descriptors", tile_abi)
        self.assertIn("@golem_tile_dma_descriptors", tile_abi)
        self.assertIn("@golem_tile_dma_descriptor_count", tile_abi)

        sharded_case = LoweringCase(
            name="route_final_consumers",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        sharded = run_pipeline(
            sharded_case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        consumer = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-extract-tile-module=tile-id=0",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-finalize-tile-runtime-graph",
            ],
            input=sharded,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(consumer.returncode, 0, consumer.stderr)
        route_input_owners = {
            int(owner)
            for owner in re.findall(
                r"#sculptor\.tile_memory_owner<id = (\d+) : i64[^>]*"
                r"kind = route_input",
                consumer.stdout,
            )
        }
        self.assertEqual(len(route_input_owners), 3)
        for owner in route_input_owners:
            self.assertRegex(
                consumer.stdout,
                rf"kind = final_consumer_complete[^>]*ownerId = {owner} : i64",
            )
            self.assertRegex(
                consumer.stdout,
                rf"kind = owner_release[^>]*ownerId = {owner} : i64",
            )
        route_offsets = [
            int(offset)
            for owner, offset in re.findall(
                r"#sculptor\.tile_memory_lifetime<[^>]*storage = workspace, "
                r"ownerId = (\d+) : i64[^>]*offset = (\d+) : i64",
                consumer.stdout,
            )
            if int(owner) in route_input_owners
        ]
        self.assertEqual(len(route_offsets), 3)
        self.assertEqual(len(set(route_offsets)), 3)
        self.assertGreaterEqual(
            consumer.stdout.count("kind = dma_completion"), 3
        )
        self.assertIn("kind = assembly_join", consumer.stdout)

        route_chain_case = LoweringCase(
            name="ordered_route_buffer_reuse",
            model_factory=ThreeLayerRouteLinear,
            input_factory=lambda: (torch.ones(1, 8),),
            array_rows=8,
            array_cols=2,
        )
        route_chain = run_pipeline(
            route_chain_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=spread mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4 array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=snake mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        active_tiles = sorted(
            {int(tile) for tile in re.findall(r"module @tile_(\d+)", route_chain)}
        )
        found_ordered_route_reuse = False
        for tile in active_tiles:
            finalized = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    "-",
                    f"--sculptor-extract-tile-module=tile-id={tile}",
                    "--sculptor-materialize-tile-runtime-graph",
                    "--sculptor-finalize-tile-runtime-graph",
                ],
                input=route_chain,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(finalized.returncode, 0, finalized.stderr)
            owner_kinds = {
                int(owner): kind
                for owner, kind in re.findall(
                    r"#sculptor\.tile_memory_owner<id = (\d+) : i64[^>]*"
                    r"kind = (\w+)",
                    finalized.stdout,
                )
            }
            route_lifetimes = {
                int(lifetime): (int(owner), int(offset))
                for lifetime, owner, offset in re.findall(
                    r"#sculptor\.tile_memory_lifetime<id = (\d+) : i64, "
                    r"subjectKind = owner, storage = workspace, "
                    r"ownerId = (\d+) : i64[^>]*offset = (\d+) : i64",
                    finalized.stdout,
                )
                if owner_kinds.get(int(owner)) in {"route_input", "route_output"}
            }
            for left, right, relation in re.findall(
                r"#sculptor\.tile_memory_interference<id = \d+ : i64, "
                r"leftLifetimeId = (\d+) : i64, "
                r"rightLifetimeId = (\d+) : i64, relation = (\w+)>",
                finalized.stdout,
            ):
                left_id = int(left)
                right_id = int(right)
                if left_id not in route_lifetimes or right_id not in route_lifetimes:
                    continue
                left_owner, left_offset = route_lifetimes[left_id]
                right_owner, right_offset = route_lifetimes[right_id]
                if (
                    relation in {"before", "after"}
                    and left_offset == right_offset
                    and owner_kinds[left_owner] != owner_kinds[right_owner]
                ):
                    found_ordered_route_reuse = True
        self.assertTrue(found_ordered_route_reuse)

    def test_global_scratchpad_selection_is_deterministic(self):
        case = LoweringCase(
            name="global_scratchpad_selection",
            model_factory=ForkedElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        outlined = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work=parallel-workers=1",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=1 mesh-cols=1 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )

        def plan(capacity: int) -> str:
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    "-",
                    "--sculptor-extract-tile-module=tile-id=0",
                    "--sculptor-materialize-tile-runtime-graph",
                    "--sculptor-plan-tile-scratchpad="
                    f"bytes={capacity} alignment=64",
                    "--sculptor-finalize-tile-runtime-graph",
                ],
                input=outlined,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout

        constrained = plan(128)
        self.assertEqual(constrained, plan(128))
        candidates = set(
            re.findall(
                r"#sculptor\.tile_scratchpad_candidate<[^>]+>", constrained
            )
        )
        allocations = set(
            re.findall(
                r"#sculptor\.tile_scratchpad_allocation<[^>]+>", constrained
            )
        )
        self.assertEqual(len(candidates), 4)
        self.assertEqual(len(allocations), 2)
        selected_ids = {
            int(candidate_id)
            for candidate_id in re.findall(
                r"#sculptor\.tile_scratchpad_candidate<id = (\d+) : i64[^>]*"
                r"selected = true>",
                constrained,
            )
        }
        self.assertEqual(selected_ids, {0, 1})
        self.assertIn("sculptor.memory.scratchpad_capacity = 128 : i64", constrained)
        self.assertIn("sculptor.runtime.scratchpad_required_bytes = 128 : i64", constrained)
        self.assertIn("scratchpadBytes = 128 : i64", constrained)
        self.assertIn("workspaceBytes = 128 : i64", constrained)

        full = plan(256)
        full_allocations = set(
            re.findall(
                r"#sculptor\.tile_scratchpad_allocation<[^>]+>", full
            )
        )
        self.assertEqual(len(full_allocations), 4)
        offsets = sorted(
            int(offset)
            for offset in re.findall(
                r"#sculptor\.tile_scratchpad_allocation<[^>]*"
                r"offset = (\d+) : i64",
                full,
            )
        )
        self.assertEqual(offsets, [0, 0, 128, 128])
        self.assertIn("sculptor.runtime.scratchpad_required_bytes = 256 : i64", full)
        self.assertIn("workspaceBytes = 0 : i64", full)
        self.assertIn("scratchpadBytes = 256 : i64", full)

        tile_abi = emit_outlined_tile_abi(
            outlined,
            0,
            ("--sculptor-plan-tile-scratchpad=bytes=256 alignment=64",),
        )
        self.assertIn("@golem_tile_scratchpad_required_bytes", tile_abi)
        self.assertIn("@golem_tile_dma_descriptor_count", tile_abi)
        translate = ANALOG_ROOT / "build" / "llvm-project" / "bin" / "mlir-translate"
        translated = subprocess.run(
            [str(translate), "--mlir-to-llvmir"],
            input=tile_abi,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(translated.returncode, 0, translated.stderr)

    def test_elementwise_in_place_alias_reaches_bufferized_routines(self):
        case = LoweringCase(
            name="elementwise_in_place_alias",
            model_factory=LongElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        producing_passes = [
            "--sculptor-expand-digital-work=parallel-workers=1",
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=recursive-fork-join "
                "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                "array-rows=8 array-cols=2"
            ),
            (
                "--sculptor-place-logical-tiles="
                "schedule=greedy mesh-rows=1 mesh-cols=1 "
                "arrays-per-core=4"
            ),
            "--sculptor-outline-tile-routines",
        ]
        consuming_passes = [
            "--sculptor-extract-tile-module=tile-id=0",
            "--sculptor-materialize-tile-runtime-graph",
            "--sculptor-finalize-tile-runtime-graph",
        ]
        finalized = run_serialized_pipeline(
            case, producing_passes, consuming_passes
        )
        self.assertEqual(
            finalized.count("#sculptor.tile_memory_in_place_alias<"), 3
        )
        self.assertEqual(finalized.count("relation = in_place_alias"), 3)
        self.assertIn("workspaceBytes = 128 : i64", finalized)
        self.assertIn("intermediateBytes = 512 : i64", finalized)
        self.assertIn("reusableBytes = 384 : i64", finalized)
        self.assertEqual(
            len(
                re.findall(
                    r"storage = workspace,.*?offset = 0 : i64",
                    finalized,
                    re.DOTALL,
                )
            ),
            4,
        )

        result = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--empty-tensor-to-alloc-tensor",
                "--one-shot-bufferize=bufferize-function-boundaries "
                "function-boundary-type-conversion=identity-layout-map",
                "--buffer-results-to-out-params=hoist-static-allocs",
                "--convert-bufferization-to-memref",
                "--sculptor-bind-tile-routine-destinations",
                "--buffer-hoisting",
                "--buffer-loop-hoisting",
                "--buffer-deallocation-pipeline",
                "--optimize-allocation-liveness",
                "--sculptor-audit-tile-bufferization=strict=true",
            ],
            input=finalized,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("memref.alloc", result.stdout)
        self.assertNotIn("memref.copy", result.stdout)
        self.assertIn("sculptor.memory.destination_bound", result.stdout)

    def test_elementwise_in_place_alias_rejects_pending_fan_out_readers(self):
        case = LoweringCase(
            name="elementwise_in_place_fan_out",
            model_factory=SharedElementwiseFanOut,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        finalized = run_serialized_pipeline(
            case,
            [
                "--sculptor-expand-digital-work=parallel-workers=1",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=1 mesh-cols=1 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
            [
                "--sculptor-extract-tile-module=tile-id=0",
                "--sculptor-materialize-tile-runtime-graph",
                "--sculptor-finalize-tile-runtime-graph",
            ],
        )

        computation_readers: dict[int, int] = {}
        for owner, effect in re.findall(
            r"#sculptor\.tile_memory_binding<[^>]*input = true, "
            r"ownerId = (\d+) : i64,[^>]*effect = (read|read_write)>",
            finalized,
        ):
            owner_id = int(owner)
            computation_readers[owner_id] = (
                computation_readers.get(owner_id, 0) + 1
            )
        fan_out_owners = {
            owner for owner, readers in computation_readers.items() if readers > 1
        }
        self.assertTrue(fan_out_owners)

        aliased_inputs = {
            int(owner)
            for owner in re.findall(
                r"#sculptor\.tile_memory_in_place_alias<[^>]*"
                r"inputOwnerId = (\d+) : i64",
                finalized,
            )
        }
        self.assertTrue(aliased_inputs)
        self.assertTrue(fan_out_owners.isdisjoint(aliased_inputs))

    def test_safe_producer_consumer_fusion_removes_internal_boundaries(self):
        case = LoweringCase(
            name="producer_consumer_fusion",
            model_factory=LongElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        prefix = [
            "--sculptor-expand-digital-work=parallel-workers=1",
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "mesh-rows=1 mesh-cols=1 arrays-per-core=4 "
                "array-rows=8 array-cols=2"
            ),
            (
                "--sculptor-place-logical-tiles="
                "schedule=greedy mesh-rows=1 mesh-cols=1 "
                "arrays-per-core=4"
            ),
        ]
        baseline = run_pipeline(
            case, [*prefix, "--sculptor-outline-tile-routines"]
        )
        fused = run_pipeline(
            case,
            [
                *prefix,
                "--sculptor-outline-tile-routines="
                "fuse-producer-consumer=true",
            ],
        )

        self.assertEqual(
            len(re.findall(r"func.func private @routine_", baseline)), 5
        )
        self.assertEqual(
            len(re.findall(r"func.func private @routine_", fused)), 1
        )
        self.assertEqual(baseline.count("linalg.generic"), 5)
        self.assertEqual(fused.count("linalg.generic"), 1)
        self.assertNotIn(
            "sculptor.memory.fused_producer_consumer_count", baseline
        )
        self.assertIn(
            "sculptor.memory.fused_producer_consumer_count = 4 : i64",
            fused,
        )
        self.assertIn(
            "sculptor.memory.fused_producer_consumer_boundary_bytes = "
            "512 : i64",
            fused,
        )
        self.assertNotIn("#sculptor.tile_routine_binding<", fused)
        self.assertNotIn("#sculptor.tile_routine_route<", fused)
        self.assertIn("sculptor.memory.plan_version = 3 : i64", fused)
        self.assertIn("sculptor.memory.owners", fused)
        self.assertIn("sculptor.memory.views", fused)
        self.assertIn("@golem_tile_dispatch_tasks", emit_outlined_tile_abi(fused, 0))

    def test_producer_consumer_fusion_preserves_analog_and_assembly_boundaries(
        self,
    ):
        analog = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines="
                "fuse-producer-consumer=true",
            ],
        )
        self.assertIn(
            "sculptor.memory.fused_producer_consumer_count = 0 : i64",
            analog,
        )
        self.assertEqual(analog.count("sculptor.array.set"), 4)
        self.assertEqual(analog.count("sculptor.array.execute"), 4)
        self.assertEqual(analog.count("sculptor.array.store"), 4)

        sharded_case = LoweringCase(
            name="fusion_assembly_boundary",
            model_factory=ElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8), torch.ones(4, 8)),
            minimum_matrix_setups=0,
        )
        sharded_prefix = [
            "--sculptor-expand-digital-work="
            "parallel-workers=4 dataflow=sharded",
            "--sculptor-build-ra-tree",
            (
                "--sculptor-plan-mapping="
                "strategies=recursive-fork-join "
                "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                "array-rows=8 array-cols=2"
            ),
            (
                "--sculptor-place-logical-tiles="
                "schedule=greedy mesh-rows=4 mesh-cols=4 "
                "arrays-per-core=4"
            ),
        ]
        baseline = run_pipeline(
            sharded_case,
            [*sharded_prefix, "--sculptor-outline-tile-routines"],
        )
        fused = run_pipeline(
            sharded_case,
            [
                *sharded_prefix,
                "--sculptor-outline-tile-routines="
                "fuse-producer-consumer=true",
            ],
        )
        self.assertIn(
            "sculptor.memory.fused_producer_consumer_count = 0 : i64",
            fused,
        )
        self.assertEqual(
            baseline.count("#sculptor.tile_routine_route<"),
            fused.count("#sculptor.tile_routine_route<"),
        )
        self.assertEqual(
            len(re.findall(r"func.func private @routine_", baseline)),
            len(re.findall(r"func.func private @routine_", fused)),
        )
        self.assertIn("#sculptor.tile_memory_assembly<", fused)

    def test_nested_assembly_writes_directly_into_runtime_destination(self):
        outlined = run_ir_pipeline(
            NESTED_ASSEMBLY_FIXTURE,
            [
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=2 mesh-cols=2 "
                    "arrays-per-core=1 verify-placement"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "outlined.mlir"
            source.write_text(outlined)
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    str(source),
                    "--sculptor-extract-tile-module=tile-id=0",
                    "--sculptor-materialize-tile-runtime-graph",
                    "--sculptor-finalize-tile-runtime-graph",
                    "--sculptor-lower-golem-to-llvm-shims",
                    "--canonicalize",
                    "--cse",
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=bufferize-function-boundaries "
                    "function-boundary-type-conversion=identity-layout-map",
                    "--buffer-results-to-out-params=hoist-static-allocs",
                    "--convert-bufferization-to-memref",
                    "--sculptor-bind-tile-routine-destinations",
                    "--buffer-hoisting",
                    "--buffer-loop-hoisting",
                    "--buffer-deallocation-pipeline",
                    "--optimize-allocation-liveness",
                    "--sculptor-audit-tile-bufferization=strict=true",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        assembly_routine = result.stdout.split(
            "func.func private @routine_4", 1
        )[1].split("func.func private @generate_task_graph", 1)[0]
        self.assertEqual(
            assembly_routine.count("sculptor.memory.direct_nested_assembly"),
            2,
        )
        self.assertNotIn("memref.alloc", assembly_routine)
        self.assertNotRegex(
            assembly_routine,
            r"memref.copy %collapse_shape[^\n]*%subview",
        )
        self.assertIn("unplanned_allocation_count = 0 : i64", result.stdout)
        self.assertIn("unplanned_copy_count = 0 : i64", result.stdout)

    def test_sharded_concat_transposes_directly_without_assembly_buffer(self):
        outlined = run_ir_pipeline(
            ASSEMBLED_TRANSPOSE_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=2 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=1 "
                    "array-rows=8 array-cols=8 verify-plan"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=2 mesh-cols=2 "
                    "arrays-per-core=1 verify-placement"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "outlined.mlir"
            source.write_text(outlined)
            result = subprocess.run(
                [
                    str(DEFAULT_SCULPTOR_OPT),
                    str(source),
                    "--sculptor-extract-tile-module=tile-id=0",
                    "--sculptor-materialize-tile-runtime-graph",
                    "--sculptor-finalize-tile-runtime-graph",
                    "--sculptor-lower-golem-to-llvm-shims",
                    "--canonicalize",
                    "--cse",
                    "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=bufferize-function-boundaries "
                    "function-boundary-type-conversion=identity-layout-map",
                    "--buffer-results-to-out-params=hoist-static-allocs",
                    "--convert-bufferization-to-memref",
                    "--sculptor-bind-tile-routine-destinations",
                    "--buffer-hoisting",
                    "--buffer-loop-hoisting",
                    "--buffer-deallocation-pipeline",
                    "--optimize-allocation-liveness",
                    "--sculptor-audit-tile-bufferization=strict=true",
                    "--sculptor-report-tile-memory=stage=regression",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        first_routine = result.stdout.split(
            "func.func private @routine_0", 1
        )[1].split("func.func private @routine_2", 1)[0]
        self.assertIn("sculptor.memory.assembled_transpose_direct", first_routine)
        self.assertEqual(
            first_routine.count("sculptor.memory.direct_assembled_transpose"),
            2,
        )
        self.assertNotIn("memref.alloc", first_routine)
        self.assertNotIn("memref.copy", first_routine)
        self.assertIn("routineTemporaryPeakBytes = 0 : i64", result.stdout)
        self.assertIn("unplanned_allocation_count = 0 : i64", result.stdout)
        self.assertIn("unplanned_copy_count = 0 : i64", result.stdout)

    def test_segmented_transpose_plan_reaches_the_tile_abi(self):
        outlined = run_ir_pipeline(
            SEGMENTED_TRANSPOSE_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        movements = set(
            re.findall(
                r"#sculptor\.tile_memory_movement<[^>]*"
                r"mode = segmented[^>]+>",
                outlined,
            )
        )
        segments = set(
            re.findall(r"#sculptor\.tile_memory_segment<[^>]+>", outlined)
        )
        self.assertEqual(len(movements), 4)
        self.assertEqual(len(segments), 32)
        self.assertNotIn("mode = packed", outlined)
        self.assertIn(
            "sourceByteOffset = 0 : i64, "
            "destinationByteOffset = 0 : i64, byteSize = 8 : i64",
            outlined,
        )
        self.assertIn(
            "sourceByteOffset = 56 : i64, "
            "destinationByteOffset = 248 : i64, byteSize = 8 : i64",
            outlined,
        )

        tile_abi = emit_outlined_tile_abi(outlined, 0)
        for symbol in (
            "@__golem_tile_segmented_movements",
            "@__golem_tile_memory_segments",
            "@golem_tile_segmented_movements",
            "@golem_tile_segmented_movement_count",
            "@golem_tile_memory_segments",
            "@golem_tile_memory_segment_count",
        ):
            self.assertIn(symbol, tile_abi)
        self.assertIn("!llvm.array<3 x struct<", tile_abi)
        self.assertIn("!llvm.array<24 x struct<", tile_abi)
        self.assertNotIn("#sculptor.", tile_abi)
        translate = ANALOG_ROOT / "build" / "llvm-project" / "bin" / "mlir-translate"
        translated = subprocess.run(
            [str(translate), "--mlir-to-llvmir"],
            input=tile_abi,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(translated.returncode, 0, translated.stderr)
        self.assertIn("@golem_tile_memory_segments", translated.stdout)

    def test_segmented_plan_rejects_malformed_tables(self):
        outlined = run_ir_pipeline(
            SEGMENTED_TRANSPOSE_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        extracted = subprocess.run(
            [
                str(DEFAULT_SCULPTOR_OPT),
                "-",
                "--sculptor-extract-tile-module=tile-id=0",
            ],
            input=outlined,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(extracted.returncode, 0, extracted.stderr)
        standalone = extracted.stdout
        segment_records = re.findall(
            r"#sculptor\.tile_memory_segment<[^>]+>", standalone
        )
        self.assertGreaterEqual(len(segment_records), 2)
        second_overlap = re.sub(
            r"sourceByteOffset = [0-9]+",
            "sourceByteOffset = 0",
            segment_records[1],
            count=1,
        )
        second_ordinal = re.sub(
            r"ordinal = 1",
            "ordinal = 2",
            segment_records[1],
            count=1,
        )
        corruptions = {
            "empty": (
                re.sub(
                    r"sculptor\.memory\.segments = \[[^]]*\]",
                    "sculptor.memory.segments = []",
                    standalone,
                    count=1,
                ),
                "segmented movement requires at least two nonempty segments",
            ),
            "out_of_range": (
                re.sub(
                    r"(#sculptor\.tile_memory_segment<[^>]*"
                    r"sourceByteOffset = )0",
                    r"\g<1>4096",
                    standalone,
                    count=1,
                ),
                "memory segment lies outside its endpoint view geometry",
            ),
            "wrong_total": (
                re.sub(
                    r"(#sculptor\.tile_memory_segment<[^>]*byteSize = )8",
                    r"\g<1>4",
                    standalone,
                    count=1,
                ),
                "segmented movement byte count does not match its movement",
            ),
            "overlap": (
                standalone.replace(segment_records[1], second_overlap, 1),
                "segmented movement contains overlapping endpoint ranges",
            ),
            "ordinal": (
                standalone.replace(segment_records[1], second_ordinal, 1),
                "memory segment ordinals are not contiguous",
            ),
        }
        for name, (corrupted, diagnostic) in corruptions.items():
            with self.subTest(name=name):
                result = subprocess.run(
                    [
                        str(DEFAULT_SCULPTOR_OPT),
                        "-",
                        "--sculptor-materialize-tile-runtime-graph",
                    ],
                    input=corrupted,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_large_strided_transfer_uses_packed_fallback(self):
        outlined = run_ir_pipeline(
            PACKED_TRANSPOSE_FIXTURE,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=2 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=2 mesh-cols=2 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=2 mesh-cols=2 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        packed = set(
            re.findall(
                r"#sculptor\.tile_memory_movement<[^>]*mode = packed[^>]+>",
                outlined,
            )
        )
        self.assertEqual(len(packed), 2)
        self.assertNotIn("mode = segmented", outlined)
        self.assertNotIn("#sculptor.tile_memory_segment<", outlined)

        tile_abi = emit_outlined_tile_abi(outlined, 0)
        self.assertNotIn("@__golem_tile_segmented_movements", tile_abi)
        self.assertNotIn("@__golem_tile_memory_segments", tile_abi)
        self.assertNotIn("#sculptor.", tile_abi)

    def test_multi_producer_elementwise_consumer_has_exact_edges(self):
        case = LoweringCase(
            name="multi_producer_shard_dataflow",
            model_factory=ForkedElementwiseChain,
            input_factory=lambda: (torch.ones(4, 8),),
            minimum_matrix_setups=0,
        )
        prefix = [
            "--sculptor-expand-digital-work="
            "parallel-workers=4 dataflow=sharded",
            "--sculptor-build-ra-tree",
        ]
        lowered = run_pipeline(case, prefix)

        self.assertIn("sculptor.mapping.shard_edge_count = 16", lowered)
        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 1", lowered
        )
        exact_edges = set(
            re.findall(
                r"#sculptor\.mapping_work_unit_edge<([^>]*)>", lowered
            )
        )
        self.assertEqual(len(exact_edges), 16)

        outlined = run_pipeline(
            case,
            [
                *prefix,
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("module @tile_", outlined)
        self.assertIn("#sculptor.tile_routine_route<", outlined)

    def test_communication_aware_tiling_preserves_transpose_shards(self):
        case = LoweringCase(
            name="communication_aware_transpose_sandwich",
            model_factory=TransposeSandwich,
            input_factory=lambda: (torch.ones(8, 8),),
            minimum_matrix_setups=0,
        )
        dimension_first = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=dimension-first",
                "--sculptor-build-ra-tree",
            ],
        )
        communication_aware = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=communication-aware",
                "--sculptor-build-ra-tree",
            ],
        )

        self.assertIn(
            "sculptor.mapping.shard_edge_count = 20", dimension_first
        )
        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 1", dimension_first
        )
        self.assertIn(
            'sculptor.mapping.digital_tiling_policy = "communication-aware"',
            communication_aware,
        )
        self.assertIn(
            "sculptor.mapping.shard_edge_count = 8", communication_aware
        )
        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 1",
            communication_aware,
        )
        exact_edges = set(
            re.findall(
                r"#sculptor\.mapping_work_unit_edge<([^>]*)>",
                communication_aware,
            )
        )
        self.assertEqual(len(exact_edges), 8)
        self.assertEqual(
            sum(
                "sourceOperationId = 0" in edge
                and "targetOperationId = 1" in edge
                for edge in exact_edges
            ),
            4,
        )
        self.assertEqual(
            sum(
                "sourceOperationId = 1" in edge
                and "targetOperationId = 2" in edge
                for edge in exact_edges
            ),
            4,
        )
        self.assertIn("resultSizes = [8, 2]", communication_aware)
        self.assertIn("resultSizes = [2, 8]", communication_aware)

        outlined = run_pipeline(
            case,
            [
                "--sculptor-expand-digital-work="
                "parallel-workers=4 dataflow=sharded "
                "tiling-policy=communication-aware",
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("tensor.extract_slice", outlined)
        self.assertNotIn("sculptor.memory.zero_copy_view", outlined)

    def test_communication_aware_sharding_crosses_dynamic_matmul_reduction(self):
        case = LoweringCase(
            name="communication_aware_dynamic_matmul",
            model_factory=DynamicMatmulJoin,
            input_factory=lambda: (torch.ones(8, 8), torch.ones(8, 8)),
            minimum_matrix_setups=0,
        )
        passes = [
            "--sculptor-expand-digital-work="
            "parallel-workers=4 dataflow=sharded "
            "tiling-policy=communication-aware",
            "--sculptor-build-ra-tree",
        ]
        lowered = run_pipeline(case, passes)

        self.assertIn(
            "sculptor.mapping.assembly_boundary_count = 1", lowered
        )
        self.assertIn("sculptor.mapping.shard_edge_count = 28", lowered)
        self.assertIn(
            "sculptor.mapping.expanded_digital_work_unit_count = 24", lowered
        )
        matmul_line = next(
            line for line in lowered.splitlines() if "linalg.matmul {" in line
        )
        self.assertEqual(matmul_line.count("#sculptor.mapping_work_unit<"), 4)
        self.assertEqual(
            set(re.findall(r"resultSizes = \[([^]]+)\]", matmul_line)),
            {"4, 4"},
        )

        exact_edges = set(
            re.findall(
                r"#sculptor\.mapping_work_unit_edge<([^>]*)>", lowered
            )
        )
        matmul_inputs = [
            edge for edge in exact_edges if "targetOperationId = 5" in edge
        ]
        self.assertEqual(len(matmul_inputs), 20)
        self.assertTrue(
            all("byteSize = 64" in edge for edge in matmul_inputs)
        )

        outlined = run_pipeline(
            case,
            [
                *passes,
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=8"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )
        self.assertIn("tensor<4x4xf32>", outlined)
        self.assertIn("mode = segmented", outlined)
        self.assertNotIn("mode = packed", outlined)

    def test_consumer_bound_fill_colocates_sharded_work_units(self):
        case = LoweringCase(
            name="consumer_bound_fill_layer_norm",
            model_factory=LayerNormOnly,
            input_factory=lambda: (torch.ones(1, 4, 8),),
            minimum_matrix_setups=0,
        )
        placed = run_pipeline(
            case,
            [
                "--sculptor-canonicalize-layers",
                "--sculptor-extract-layers",
                "--sculptor-convert-layers",
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=4 dataflow=sharded "
                    "shard-propagation-depth=0 "
                    "require-complete-shard-chain=false"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=recursive-fork-join,consumer-bound-fill "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        report = build_ra_tree_report(placed)["functions"][0]
        operations = {
            operation["id"]: operation
            for operation in report["operations"]
        }
        nodes = {node["id"]: node for node in report["tree"]["nodes"]}
        work_units = {unit["id"]: unit for unit in report["tree"]["work_units"]}
        assignments = {
            (
                assignment["operation_id"],
                nodes[assignment["leaf_id"]]["work_unit_id"],
            ): assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
        }
        consumers = {}
        for edge in report["edges"]:
            consumers.setdefault(edge["source"], set()).add(edge["target"])

        eligible_fill_ids = [
            operation_id
            for operation_id, operation in operations.items()
            if operation["name"] == "linalg.fill"
            and len(consumers.get(operation_id, set())) == 1
        ]
        self.assertEqual(len(eligible_fill_ids), 1)
        for fill_id in eligible_fill_ids:
            consumer_id = next(iter(consumers[fill_id]))
            fill_units = [
                unit
                for unit in work_units.values()
                if unit["operation_id"] == fill_id
            ]
            consumer_units = [
                unit
                for unit in work_units.values()
                if unit["operation_id"] == consumer_id
            ]
            self.assertEqual(len(fill_units), 4)
            self.assertEqual(len(consumer_units), 4)
            for fill_unit in fill_units:
                matching_consumers = [
                    unit for unit in consumer_units
                    if unit["result_offsets"] == fill_unit["result_offsets"]
                    and unit["result_sizes"] == fill_unit["result_sizes"]
                ]
                self.assertEqual(len(matching_consumers), 1)
                consumer_unit = matching_consumers[0]
                self.assertEqual(
                    assignments[(fill_id, fill_unit["id"])]["tile_id"],
                    assignments[(consumer_id, consumer_unit["id"])]["tile_id"],
                )

    def test_balanced_reduction_tree_reaches_outlined_routines(self):
        lowered = run_pipeline(
            self.linear_case,
            [
                *linear_prefix(),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 reduction-tree=balanced "
                    "reduction-fan-in=2 reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
                "--sculptor-outline-tile-routines",
            ],
        )

        self.assertIn("sculptor.mapping.reduction_tree_count = 1", lowered)
        self.assertIn("sculptor.mapping.reduction_node_count = 3", lowered)
        self.assertIn("sculptor.mapping.maximum_reduction_fan_in = 2", lowered)
        node_ids = {
            int(value)
            for value in re.findall(
                r"sculptor\.mapping\.reduction_node_id = (\d+)", lowered
            )
        }
        self.assertEqual(node_ids, {0, 1, 2})
        self.assertIn("sculptor.task.reduction_tree_id", lowered)
        self.assertIn("sculptor.task.reduction_level", lowered)
        self.assertIn("sculptor.task.reduction_width", lowered)
        materialized = materialize_outlined_tiles(lowered)
        self.assertTrue(
            any('task_kind = "digital.reduction"' in ir for ir in materialized)
        )

    def test_packed_balanced_reduction_keeps_ready_mvm_bodies_parallel(self):
        case = LoweringCase(
            name="packed_three_way_reduction",
            model_factory=ThreeWayReductionLinear,
            input_factory=lambda: (torch.ones(1, 6),),
            array_rows=8,
            array_cols=2,
        )
        placed = run_pipeline(
            case,
            [
                *linear_prefix(array_cols=2),
                (
                    "--sculptor-expand-digital-work="
                    "parallel-workers=1 reduction-tree=balanced "
                    "reduction-fan-in=2 reduction-min-width=3"
                ),
                "--sculptor-build-ra-tree",
                (
                    "--sculptor-plan-mapping="
                    "strategies=setup-first,recursive-fork-join "
                    "mvm-body-policy=packed "
                    "digital-scheduling-policy=progressive "
                    "mesh-rows=4 mesh-cols=4 arrays-per-core=4 "
                    "array-rows=8 array-cols=2"
                ),
                (
                    "--sculptor-place-logical-tiles="
                    "schedule=greedy mesh-rows=4 mesh-cols=4 "
                    "arrays-per-core=4"
                ),
            ],
        )
        report = build_ra_tree_report(placed)["functions"][0]
        nodes_by_id = {node["id"]: node for node in report["tree"]["nodes"]}
        vector_ids = [
            operation["id"]
            for operation in report["operations"]
            if operation["kind"] == "vector_tile"
        ]
        mvm_ids = [
            operation["id"]
            for operation in report["operations"]
            if operation["kind"] == "physical_mvm"
        ]
        self.assertEqual(len(vector_ids), 3)
        self.assertEqual(len(mvm_ids), 3)

        leaves_by_operation = {
            node["operation_id"]: node
            for node in nodes_by_id.values()
            if node["kind"] == "leaf" and node["operation_id"] in vector_ids
        }
        body_roots = [
            nodes_by_id[leaves_by_operation[operation_id]["parent_id"]]
            for operation_id in vector_ids
        ]
        cohort_ids = {body["parent_id"] for body in body_roots}
        self.assertEqual(len(cohort_ids), 1)
        cohort = nodes_by_id[next(iter(cohort_ids))]
        self.assertEqual(cohort["kind"], "spatial_cut")

        assignments = {
            assignment["operation_id"]: assignment
            for assignment in report["plan"]["realization"]["leaf_assignments"]
            if assignment["operation_id"] in vector_ids + mvm_ids
        }
        self.assertEqual(len(assignments), 6)
        self.assertEqual(
            {assignments[operation_id]["start_ns"] for operation_id in vector_ids},
            {assignments[vector_ids[0]]["start_ns"]},
        )
        self.assertEqual(
            {assignments[operation_id]["start_ns"] for operation_id in mvm_ids},
            {assignments[mvm_ids[0]]["start_ns"]},
        )


if __name__ == "__main__":
    unittest.main()
