#ifndef SCULPTOR_NN_OPS_TD
#define SCULPTOR_NN_OPS_TD

include "mlir/IR/CommonAttrConstraints.td"
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

#endif // SCULPTOR_NN_OPS_TD
