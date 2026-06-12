#ifndef SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H
#define SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H

#include "sculptor-mlir/Dialect/Sculptor/IR/SculptorDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace sculptor {

// Records the hardware budget that later task-graph scheduling and placement
// decisions must obey.
struct ScheduleTaskGraphPass
    : public mlir::PassWrapper<ScheduleTaskGraphPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleTaskGraphPass)

  Option<int64_t> cores{
      *this, "cores",
      llvm::cl::desc("Number of available analog/digital cores"),
      llvm::cl::init(1)};

  Option<int64_t> arraysPerCore{
      *this, "arrays-per-core",
      llvm::cl::desc("Number of physical analog arrays attached to each core"),
      llvm::cl::init(1)};

  Option<std::string> topology{
      *this, "topology",
      llvm::cl::desc("Hardware topology used by the task graph scheduler"),
      llvm::cl::init("mesh")};

  Option<int64_t> meshRows{*this, "mesh-rows",
                           llvm::cl::desc("Number of rows in a mesh topology"),
                           llvm::cl::init(1)};

  Option<int64_t> meshCols{
      *this, "mesh-cols",
      llvm::cl::desc(
          "Number of columns in a mesh topology, or 0 to infer from cores"),
      llvm::cl::init(0)};

  Option<std::string> schedule{
      *this, "schedule",
      llvm::cl::desc("Registered task graph scheduling algorithm to run"),
      llvm::cl::init("simple-budget")};

  Option<std::string> placement{
      *this, "placement",
      llvm::cl::desc("Boundary-to-core placement vector for explicit "
                     "placement schedulers"),
      llvm::cl::init("")};

  ScheduleTaskGraphPass() = default;

  ScheduleTaskGraphPass(const ScheduleTaskGraphPass &pass)
      : PassWrapper(pass),
        cores(*this, "cores",
              llvm::cl::desc("Number of available analog/digital cores"),
              llvm::cl::init(1)),
        arraysPerCore(
            *this, "arrays-per-core",
            llvm::cl::desc(
                "Number of physical analog arrays attached to each core"),
            llvm::cl::init(1)),
        topology(*this, "topology",
                 llvm::cl::desc(
                     "Hardware topology used by the task graph scheduler"),
                 llvm::cl::init("mesh")),
        meshRows(*this, "mesh-rows",
                 llvm::cl::desc("Number of rows in a mesh topology"),
                 llvm::cl::init(1)),
        meshCols(*this, "mesh-cols",
                 llvm::cl::desc("Number of columns in a mesh topology, or 0 "
                                "to infer from cores"),
                 llvm::cl::init(0)),
        schedule(
            *this, "schedule",
            llvm::cl::desc("Registered task graph scheduling algorithm to run"),
            llvm::cl::init("simple-budget")),
        placement(*this, "placement",
                  llvm::cl::desc("Boundary-to-core placement vector for "
                                 "explicit placement schedulers"),
                  llvm::cl::init("")) {
    cores = pass.cores;
    arraysPerCore = pass.arraysPerCore;
    topology = pass.topology;
    meshRows = pass.meshRows;
    meshCols = pass.meshCols;
    schedule = pass.schedule;
    placement = pass.placement;
  }

  mlir::StringRef getArgument() const final {
    return "sculptor-schedule-task-graph";
  }

  mlir::StringRef getDescription() const final {
    return "Attach a hardware budget for analog task graph scheduling";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::sculptor::SculptorDialect, mlir::func::FuncDialect>();
  }

  void runOnOperation() override;
};

void registerScheduleTaskGraphPass();

} // namespace sculptor
} // namespace mlir

#endif // SCULPTOR_MLIR_DIALECT_SCULPTOR_TRANSFORMS_SCHEDULETASKGRAPH_H
