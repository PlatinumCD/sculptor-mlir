#ifndef SCULPTOR_BASE_TD
#define SCULPTOR_BASE_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Sculptor_Dialect : Dialect {
  let name = "sculptor";
  let cppNamespace = "::mlir::sculptor";
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOps();
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const;
  }];
}

class Sculptor_Type<string name, string mnem, list<Trait> traits = []>
    : TypeDef<Sculptor_Dialect, name, traits> {
  let mnemonic = mnem;
}

class Sculptor_Op<string mnem, list<Trait> traits = []>
    : Op<Sculptor_Dialect, mnem, traits> {
  let hasCustomAssemblyFormat = 0;
}

#endif // SCULPTOR_BASE_TD
