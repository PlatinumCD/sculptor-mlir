add_mlir_dialect_library(SculptorIR
  SculptorBase.cpp
  SculptorTypes.cpp
  SculptorAttrs.cpp
  SculptorOps.cpp
  Ops/SculptorNNOps.cpp
  Ops/SculptorGolemOps.cpp
  Ops/SculptorTaskGraphOps.cpp
  Ops/SculptorExecutionOps.cpp
  Ops/SculptorTaskRegionOps.cpp

  DEPENDS
  SculptorIRIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
