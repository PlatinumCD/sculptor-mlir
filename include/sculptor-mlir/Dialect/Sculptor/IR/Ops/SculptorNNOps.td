#ifndef SCULPTOR_NN_OPS_TD
#define SCULPTOR_NN_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "sculptor-mlir/Dialect/Sculptor/IR/SculptorBase.td"
include "sculptor-mlir/Dialect/Sculptor/IR/SculptorTypes.td"

class Sculptor_NNOp<string mnem, list<Trait> traits = []>
    : Sculptor_Op<"nn." # mnem, traits>;

//===----------------------------------------------------------------------===//
// sculptor.nn.linear
// Semantic neural-network linear layer.
//===----------------------------------------------------------------------===//

def Sculptor_NNLinearOp
    : Sculptor_NNOp<"linear"> {

  let summary = "Represent a canonical neural-network linear layer";

  let description = [{
    Captures the semantic input, weight, optional bias, and result tensors for
    a linear layer inside an extracted digital layer function. The verifier
    checks statically comparable shape dimensions and bias consistency while
    remaining permissive for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$has_bias
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.conv1d
// Semantic neural-network 1D convolution.
//===----------------------------------------------------------------------===//

def Sculptor_NNConv1DOp
    : Sculptor_NNOp<"conv1d"> {

  let summary = "Represent a canonical neural-network 1D convolution";

  let description = [{
    Captures the semantic activation, kernel, optional bias, spatial attributes,
    and result tensor for a 1D convolution inside an extracted digital layer
    function. The verifier checks statically comparable shape dimensions,
    spatial attributes, and bias consistency while remaining permissive for
    dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$has_bias,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.conv2d
// Semantic neural-network 2D convolution.
//===----------------------------------------------------------------------===//

def Sculptor_NNConv2DOp
    : Sculptor_NNOp<"conv2d"> {

  let summary = "Represent a canonical neural-network 2D convolution";

  let description = [{
    Captures the semantic activation, kernel, optional bias, spatial attributes,
    and result tensor for a 2D convolution inside an extracted digital layer
    function. The verifier checks statically comparable shape dimensions,
    spatial attributes, and bias consistency while remaining permissive for
    dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$has_bias,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.grouped_conv2d
// Semantic neural-network grouped 2D convolution.
//===----------------------------------------------------------------------===//

def Sculptor_NNGroupedConv2DOp
    : Sculptor_NNOp<"grouped_conv2d"> {

  let summary = "Represent a canonical neural-network grouped 2D convolution";

  let description = [{
    Captures the semantic activation, grouped kernel, optional bias, group
    count, spatial attributes, and result tensor for a grouped 2D convolution
    inside an extracted digital layer function. The verifier checks statically
    comparable shape dimensions, group/channel partitioning, spatial
    attributes, and bias consistency while remaining permissive for dynamic
    dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$has_bias,
    I64Attr:$groups,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.conv3d
// Semantic neural-network 3D convolution.
//===----------------------------------------------------------------------===//

def Sculptor_NNConv3DOp
    : Sculptor_NNOp<"conv3d"> {

  let summary = "Represent a canonical neural-network 3D convolution";

  let description = [{
    Captures the semantic activation, kernel, optional bias, spatial attributes,
    and result tensor for a 3D convolution inside an extracted digital layer
    function. The verifier checks statically comparable shape dimensions,
    spatial attributes, and bias consistency while remaining permissive for
    dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    BoolAttr:$has_bias,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$padding,
    I64ArrayAttr:$dilation
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.rnn_cell
// Semantic neural-network RNN cell.
//===----------------------------------------------------------------------===//

def Sculptor_NNRNNCellOp
    : Sculptor_NNOp<"rnn_cell", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network RNN cell";

  let description = [{
    Captures one recurrent cell step with input, previous hidden state,
    input-hidden and hidden-hidden weights, optional bias tensors, activation
    kind, and result hidden state. The verifier checks statically comparable
    shape dimensions and bias consistency while remaining permissive for
    dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h_prev,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$has_bias,
    StrAttr:$activation
  );

  let results = (outs
    AnyRankedTensor:$h
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h_prev `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.lstm_cell
// Semantic neural-network LSTM cell.
//===----------------------------------------------------------------------===//

def Sculptor_NNLSTMCellOp
    : Sculptor_NNOp<"lstm_cell", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network LSTM cell";

  let description = [{
    Captures one LSTM cell step with input, previous hidden and cell states,
    input-hidden and hidden-hidden weights, optional bias tensors, and result
    hidden and cell states. The verifier checks statically comparable
    gate-expanded dimensions and bias consistency while remaining permissive
    for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h_prev,
    AnyRankedTensor:$c_prev,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$has_bias
  );

  let results = (outs
    AnyRankedTensor:$h,
    AnyRankedTensor:$c
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h_prev `,` $c_prev `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.gru_cell
// Semantic neural-network GRU cell.
//===----------------------------------------------------------------------===//

def Sculptor_NNGRUCellOp
    : Sculptor_NNOp<"gru_cell", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network GRU cell";

  let description = [{
    Captures one GRU cell step with input, previous hidden state, input-hidden
    and hidden-hidden weights, optional bias tensors, and result hidden state.
    The verifier checks statically comparable gate-expanded dimensions and bias
    consistency while remaining permissive for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h_prev,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$has_bias
  );

  let results = (outs
    AnyRankedTensor:$h
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h_prev `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.gru
// Semantic neural-network GRU.
//===----------------------------------------------------------------------===//

def Sculptor_NNGRUOp
    : Sculptor_NNOp<"gru"> {

  let summary = "Represent a canonical neural-network GRU";

  let description = [{
    Captures the whole GRU contract with input, initial hidden state,
    recurrent weight and bias operands, static layer configuration attributes,
    sequence output, and final hidden state. The recurrent operands follow
    canonical per-layer PyTorch GRU gate ordering: reset, update,
    new/candidate. The verifier checks statically comparable layer,
    hidden-size, recurrent weight, and bias dimensions while remaining
    permissive for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    Variadic<AnyRankedTensor>:$recurrent_operands,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` `recurrent` `[` $recurrent_operands `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.lstm
// Semantic neural-network LSTM.
//===----------------------------------------------------------------------===//

def Sculptor_NNLSTMOp
    : Sculptor_NNOp<"lstm"> {

  let summary = "Represent a canonical neural-network LSTM";

  let description = [{
    Captures the whole LSTM contract with input, initial hidden and cell
    states, recurrent weight and bias operands, static layer configuration
    attributes, sequence output, final hidden state, and final cell state. The
    recurrent operands follow canonical per-layer PyTorch LSTM gate ordering:
    input, forget, candidate, output. The verifier checks statically comparable
    layer, hidden-size, recurrent weight, and bias dimensions while remaining
    permissive for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    AnyRankedTensor:$c0,
    Variadic<AnyRankedTensor>:$recurrent_operands,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn,
    AnyRankedTensor:$cn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` $c0 `,` `recurrent` `[` $recurrent_operands `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.rnn
// Semantic neural-network RNN.
//===----------------------------------------------------------------------===//

def Sculptor_NNRNNOp
    : Sculptor_NNOp<"rnn"> {

  let summary = "Represent a canonical neural-network RNN";

  let description = [{
    Captures the whole RNN contract with input, initial hidden state,
    recurrent weight and bias operands, static layer configuration attributes,
    sequence output, and final hidden state. The recurrent operands follow the
    canonical per-layer ordering described by SCULPTOR-CANONICAL.md. The verifier
    checks statically comparable layer, hidden-size, recurrent weight, and bias
    dimensions while remaining permissive for dynamic dimensions.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    Variadic<AnyRankedTensor>:$recurrent_operands,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` `recurrent` `[` $recurrent_operands `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.rnn_layer
// Semantic neural-network single RNN layer.
//===----------------------------------------------------------------------===//

def Sculptor_NNRNNLayerOp
    : Sculptor_NNOp<"rnn_layer", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network RNN layer";

  let description = [{
    Captures one full RNN layer recurrence over a batch-major sequence. The
    layer has one input sequence, the full stacked initial hidden state,
    input-hidden and hidden-hidden weights, optional bias tensors, a sequence
    output, and one rank-3 final hidden slice for this layer.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$layer_index,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.gru_layer
// Semantic neural-network single GRU layer.
//===----------------------------------------------------------------------===//

def Sculptor_NNGRULayerOp
    : Sculptor_NNOp<"gru_layer", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network GRU layer";

  let description = [{
    Captures one full GRU layer recurrence over a batch-major sequence. The
    layer has one input sequence, the full stacked initial hidden state,
    input-hidden and hidden-hidden weights, optional bias tensors, a sequence
    output, and one rank-3 final hidden slice for this layer. Gate-expanded
    operands follow PyTorch GRU gate ordering: reset, update, new/candidate.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$layer_index,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.lstm_layer
// Semantic neural-network single LSTM layer.
//===----------------------------------------------------------------------===//

def Sculptor_NNLSTMLayerOp
    : Sculptor_NNOp<"lstm_layer", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network LSTM layer";

  let description = [{
    Captures one full LSTM layer recurrence over a batch-major sequence. The
    layer has one input sequence, the full stacked initial hidden and cell
    states, input-hidden and hidden-hidden weights, optional bias tensors, a
    sequence output, and one rank-3 final hidden/cell slice for this layer.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$h0,
    AnyRankedTensor:$c0,
    AnyRankedTensor:$w_ih,
    AnyRankedTensor:$w_hh,
    Optional<AnyRankedTensor>:$b_ih,
    Optional<AnyRankedTensor>:$b_hh,
    BoolAttr:$batch_first,
    BoolAttr:$has_bias,
    I64Attr:$hidden_size,
    I64Attr:$layer_index,
    I64Attr:$num_layers
  );

  let results = (outs
    AnyRankedTensor:$output,
    AnyRankedTensor:$hn,
    AnyRankedTensor:$cn
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $h0 `,` $c0 `,` $w_ih `,` $w_hh
    (`,` $b_ih^ `,` $b_hh)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.transformer
// Semantic neural-network Transformer stack.
//===----------------------------------------------------------------------===//

def Sculptor_NNTransformerOp
    : Sculptor_NNOp<"transformer"> {

  let summary = "Represent a canonical neural-network Transformer stack";

  let description = [{
    Captures a whole encoder-decoder Transformer stack with source and target
    input sequences, an ordered variadic parameter list, static block
    configuration attributes, and one target-sequence output. Encoder
    parameters are ordered by block as self-attention QKV projection, attention
    output projection, optional attention norm affine parameters, MLP expansion,
    MLP contraction, and optional MLP norm affine parameters. Decoder parameters
    then follow by block as self-attention, cross-attention query/KV/output
    projections, their optional norms, and MLP parameters. If `has_final_norm`
    and `has_layer_norm_affine` are both true, the final parameters are encoder
    final norm scale/bias followed by decoder final norm scale/bias.
  }];

  let arguments = (ins
    AnyRankedTensor:$src,
    AnyRankedTensor:$tgt,
    Variadic<AnyRankedTensor>:$parameters,
    BoolAttr:$batch_first,
    BoolAttr:$has_projection_bias,
    BoolAttr:$has_layer_norm_affine,
    BoolAttr:$has_final_norm,
    BoolAttr:$causal,
    StrAttr:$activation,
    StrAttr:$norm_mode,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$head_dim,
    I64Attr:$mlp_hidden_size,
    I64Attr:$num_encoder_blocks,
    I64Attr:$num_decoder_blocks,
    F64Attr:$layer_norm_eps
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $src `,` $tgt `,` `parameters` `[` $parameters `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.transformer_encoder
// Semantic neural-network Transformer encoder stack.
//===----------------------------------------------------------------------===//

def Sculptor_NNTransformerEncoderOp
    : Sculptor_NNOp<"transformer_encoder"> {

  let summary = "Represent a canonical neural-network Transformer encoder stack";

  let description = [{
    Captures a one-input Transformer encoder stack with an ordered variadic
    parameter list, static block configuration attributes, and one sequence
    output. Parameters are ordered by block as self-attention QKV projection,
    attention output projection, optional attention norm affine parameters,
    MLP expansion, MLP contraction, and optional MLP norm affine parameters.
    If `has_final_norm` and `has_layer_norm_affine` are both true, the final
    parameters are final norm scale and bias.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Variadic<AnyRankedTensor>:$parameters,
    BoolAttr:$batch_first,
    BoolAttr:$has_projection_bias,
    BoolAttr:$has_layer_norm_affine,
    BoolAttr:$has_final_norm,
    BoolAttr:$causal,
    StrAttr:$activation,
    StrAttr:$norm_mode,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$head_dim,
    I64Attr:$mlp_hidden_size,
    I64Attr:$num_blocks,
    F64Attr:$layer_norm_eps
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` `parameters` `[` $parameters `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.transformer_decoder
// Semantic neural-network Transformer decoder-only stack.
//===----------------------------------------------------------------------===//

def Sculptor_NNTransformerDecoderOp
    : Sculptor_NNOp<"transformer_decoder"> {

  let summary = "Represent a canonical neural-network Transformer decoder stack";

  let description = [{
    Captures a one-input decoder-only causal Transformer stack with an ordered
    variadic parameter list, static block configuration attributes, and one
    sequence output. This op currently models the causal-only decoder case: the
    stack has self-attention and MLP parameters per block, optional final norm
    parameters, and no encoder memory or cross-attention parameter section.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Variadic<AnyRankedTensor>:$parameters,
    BoolAttr:$batch_first,
    BoolAttr:$has_projection_bias,
    BoolAttr:$has_layer_norm_affine,
    BoolAttr:$has_final_norm,
    BoolAttr:$causal,
    BoolAttr:$has_cross_attention,
    StrAttr:$decoder_kind,
    StrAttr:$activation,
    StrAttr:$norm_mode,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$head_dim,
    I64Attr:$mlp_hidden_size,
    I64Attr:$num_blocks,
    F64Attr:$layer_norm_eps
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` `parameters` `[` $parameters `]`
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// sculptor.nn.transformer_block
// Semantic neural-network Transformer block.
//===----------------------------------------------------------------------===//

def Sculptor_TransformerBlockKindEncoder
    : I32EnumAttrCase<"Encoder", 0, "encoder">;
def Sculptor_TransformerBlockKindDecoder
    : I32EnumAttrCase<"Decoder", 1, "decoder">;

def Sculptor_TransformerBlockKind
    : I32EnumAttr<"TransformerBlockKind", "Transformer block kind", [
        Sculptor_TransformerBlockKindEncoder,
        Sculptor_TransformerBlockKindDecoder
      ]> {
  let cppNamespace = "::mlir::sculptor";
}

def Sculptor_NNTransformerBlockOp
    : Sculptor_NNOp<"transformer_block", [AttrSizedOperandSegments]> {

  let summary = "Represent a canonical neural-network Transformer block";

  let description = [{
    Captures one encoder or decoder Transformer block over a batch-major
    sequence. Encoder blocks own self-attention and MLP parameters. Decoder
    blocks additionally carry an encoder memory input and cross-attention
    query, key/value, output, and normalization parameters. The last extracted
    block in a stack may also carry the stack's final normalization metadata and
    optional affine parameters. Projection biases and layer-normalization affine
    parameters are controlled independently by attributes.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Optional<AnyRankedTensor>:$memory,
    AnyRankedTensor:$qkv_weight,
    Optional<AnyRankedTensor>:$qkv_bias,
    AnyRankedTensor:$attn_output_weight,
    Optional<AnyRankedTensor>:$attn_output_bias,
    Optional<AnyRankedTensor>:$attn_norm_weight,
    Optional<AnyRankedTensor>:$attn_norm_bias,
    Optional<AnyRankedTensor>:$cross_query_weight,
    Optional<AnyRankedTensor>:$cross_query_bias,
    Optional<AnyRankedTensor>:$cross_key_value_weight,
    Optional<AnyRankedTensor>:$cross_key_value_bias,
    Optional<AnyRankedTensor>:$cross_output_weight,
    Optional<AnyRankedTensor>:$cross_output_bias,
    Optional<AnyRankedTensor>:$cross_norm_weight,
    Optional<AnyRankedTensor>:$cross_norm_bias,
    AnyRankedTensor:$mlp_up_weight,
    Optional<AnyRankedTensor>:$mlp_up_bias,
    AnyRankedTensor:$mlp_down_weight,
    Optional<AnyRankedTensor>:$mlp_down_bias,
    Optional<AnyRankedTensor>:$mlp_norm_weight,
    Optional<AnyRankedTensor>:$mlp_norm_bias,
    Optional<AnyRankedTensor>:$final_norm_weight,
    Optional<AnyRankedTensor>:$final_norm_bias,
    BoolAttr:$batch_first,
    BoolAttr:$has_projection_bias,
    BoolAttr:$has_layer_norm_affine,
    BoolAttr:$has_final_norm,
    BoolAttr:$causal,
    BoolAttr:$has_cross_attention,
    StrAttr:$activation,
    Sculptor_TransformerBlockKind:$block_kind,
    StrAttr:$norm_mode,
    I64Attr:$hidden_size,
    I64Attr:$num_heads,
    I64Attr:$head_dim,
    I64Attr:$mlp_hidden_size,
    I64Attr:$block_index,
    I64Attr:$num_blocks,
    F64Attr:$layer_norm_eps
  );

  let results = (outs
    AnyRankedTensor:$output
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,`
    `memory` `[` ($memory^)? `]` `,`
    `qkv` `[` $qkv_weight (`,` $qkv_bias^)? `]` `,`
    `attn_output` `[` $attn_output_weight (`,` $attn_output_bias^)? `]` `,`
    `attn_norm` `[` ($attn_norm_weight^ `,` $attn_norm_bias)? `]`
    `,` `cross_query` `[` ($cross_query_weight^)? (`,` $cross_query_bias^)? `]`
    `,` `cross_key_value` `[` ($cross_key_value_weight^)? (`,` $cross_key_value_bias^)? `]`
    `,` `cross_output` `[` ($cross_output_weight^)? (`,` $cross_output_bias^)? `]`
    `,` `cross_norm` `[` ($cross_norm_weight^ `,` $cross_norm_bias)? `]`
    `,` `mlp_up` `[` $mlp_up_weight (`,` $mlp_up_bias^)? `]`
    `,` `mlp_down` `[` $mlp_down_weight (`,` $mlp_down_bias^)? `]`
    `,` `mlp_norm` `[` ($mlp_norm_weight^ `,` $mlp_norm_bias)? `]`
    `,` `final_norm` `[` ($final_norm_weight^ `,` $final_norm_bias)? `]`
    `block_kind` `=` $block_kind
    attr-dict `:` functional-type(operands, results)
  }];
}

#endif // SCULPTOR_NN_OPS_TD
