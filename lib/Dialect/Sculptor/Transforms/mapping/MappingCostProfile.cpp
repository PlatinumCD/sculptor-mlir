#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingCostProfile.h"

#include "sculptor-mlir/Dialect/Sculptor/Transforms/mapping/MappingConfig.h"

#include "mlir/IR/Builders.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::sculptor::mapping;

std::string digestToHex(const std::array<uint8_t, 32> &digest) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

LogicalResult rejectUnknownKeys(const llvm::json::Object &object,
                                ArrayRef<StringRef> allowed, Operation *anchor,
                                StringRef context) {
  llvm::StringSet<> allowedSet;
  for (StringRef key : allowed)
    allowedSet.insert(key);
  for (const auto &entry : object) {
    StringRef key = entry.first;
    if (!allowedSet.contains(key))
      return anchor->emitError("unknown ") << context << " key '" << key << "'";
  }
  return success();
}

FailureOr<double> readNumber(const llvm::json::Object &object, StringRef key,
                             double defaultValue, Operation *anchor,
                             StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return defaultValue;
  std::optional<double> number = value->getAsNumber();
  if (!number) {
    anchor->emitError(context) << " field '" << key << "' must be a number";
    return failure();
  }
  if (!std::isfinite(*number) || *number < 0.0) {
    anchor->emitError(context)
        << " field '" << key << "' must be finite and nonnegative";
    return failure();
  }
  return *number;
}

FailureOr<int64_t> readPositiveInteger(const llvm::json::Object &object,
                                       StringRef key, int64_t defaultValue,
                                       Operation *anchor, StringRef context) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return defaultValue;
  std::optional<int64_t> integer = value->getAsInteger();
  if (!integer || *integer <= 0) {
    anchor->emitError(context)
        << " field '" << key << "' must be a positive integer";
    return failure();
  }
  return *integer;
}

FailureOr<TaskCostRule> parseTaskCostRule(const llvm::json::Object &object,
                                          Operation *anchor,
                                          StringRef context) {
  if (failed(rejectUnknownKeys(object,
                               {"fixed_ns", "ns_per_work_item",
                                "ns_per_input_byte", "ns_per_output_byte"},
                               anchor, context)))
    return failure();

  TaskCostRule rule;
  FailureOr<double> fixed =
      readNumber(object, "fixed_ns", 0.0, anchor, context);
  FailureOr<double> work =
      readNumber(object, "ns_per_work_item", 0.0, anchor, context);
  FailureOr<double> input =
      readNumber(object, "ns_per_input_byte", 0.0, anchor, context);
  FailureOr<double> output =
      readNumber(object, "ns_per_output_byte", 0.0, anchor, context);
  if (failed(fixed) || failed(work) || failed(input) || failed(output))
    return failure();
  rule.fixedNs = *fixed;
  rule.nsPerWorkItem = *work;
  rule.nsPerInputByte = *input;
  rule.nsPerOutputByte = *output;
  return rule;
}

DictionaryAttr serializeTaskRule(MLIRContext *context, StringRef kind,
                                 const TaskCostRule &rule) {
  Builder builder(context);
  SmallVector<NamedAttribute> fields;
  if (!kind.empty())
    fields.push_back(builder.getNamedAttr("kind", builder.getStringAttr(kind)));
  fields.push_back(
      builder.getNamedAttr("fixed_ns", builder.getF64FloatAttr(rule.fixedNs)));
  fields.push_back(builder.getNamedAttr(
      "ns_per_work_item", builder.getF64FloatAttr(rule.nsPerWorkItem)));
  fields.push_back(builder.getNamedAttr(
      "ns_per_input_byte", builder.getF64FloatAttr(rule.nsPerInputByte)));
  fields.push_back(builder.getNamedAttr(
      "ns_per_output_byte", builder.getF64FloatAttr(rule.nsPerOutputByte)));
  return DictionaryAttr::get(context, fields);
}

FailureOr<double> getF64(DictionaryAttr attr, StringRef name,
                         Operation *anchor) {
  auto value = attr.getAs<FloatAttr>(name);
  if (!value) {
    anchor->emitError("mapping cost profile is missing floating-point field '")
        << name << "'";
    return failure();
  }
  double result = value.getValueAsDouble();
  if (!std::isfinite(result) || result < 0.0) {
    anchor->emitError("mapping cost profile field '")
        << name << "' must be finite and nonnegative";
    return failure();
  }
  return result;
}

FailureOr<TaskCostRule> deserializeTaskRule(DictionaryAttr attr,
                                            Operation *anchor) {
  FailureOr<double> fixed = getF64(attr, "fixed_ns", anchor);
  FailureOr<double> work = getF64(attr, "ns_per_work_item", anchor);
  FailureOr<double> input = getF64(attr, "ns_per_input_byte", anchor);
  FailureOr<double> output = getF64(attr, "ns_per_output_byte", anchor);
  if (failed(fixed) || failed(work) || failed(input) || failed(output))
    return failure();
  return TaskCostRule{*fixed, *work, *input, *output};
}

FailureOr<int64_t> getPositiveI64(DictionaryAttr attr, StringRef name,
                                  Operation *anchor) {
  auto value = attr.getAs<IntegerAttr>(name);
  if (!value || value.getInt() <= 0) {
    anchor->emitError("mapping cost profile field '")
        << name << "' must be a positive integer";
    return failure();
  }
  return value.getInt();
}

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

LogicalResult MappingCostProfile::verify(const MappingHardwareModel &hardware,
                                         Operation *anchor) const {
  if (schemaVersion != 1)
    return anchor->emitError("unsupported mapping cost profile schema version ")
           << schemaVersion;
  if (name.empty())
    return anchor->emitError("mapping cost profile name must not be empty");
  if (source.empty())
    return anchor->emitError("mapping cost profile source must not be empty");
  if (clockFrequencyHz <= 0 || clockFrequencyHz != hardware.clockFrequencyHz)
    return anchor->emitError(
               "mapping cost profile clock frequency must match hardware: ")
           << clockFrequencyHz << " versus " << hardware.clockFrequencyHz;
  if (network.wordBits <= 0 || network.wordBits != hardware.networkWordBits)
    return anchor->emitError(
               "mapping cost profile network word width must match hardware: ")
           << network.wordBits << " versus " << hardware.networkWordBits;

  auto verifyValue = [&](double value, StringRef name) -> LogicalResult {
    if (!std::isfinite(value) || value < 0.0)
      return anchor->emitError("mapping cost profile field '")
             << name << "' must be finite and nonnegative";
    return success();
  };
  auto verifyRule = [&](const TaskCostRule &rule,
                        StringRef prefix) -> LogicalResult {
    return success(
        succeeded(verifyValue(rule.fixedNs, (prefix + ".fixed_ns").str())) &&
        succeeded(verifyValue(rule.nsPerWorkItem,
                              (prefix + ".ns_per_work_item").str())) &&
        succeeded(verifyValue(rule.nsPerInputByte,
                              (prefix + ".ns_per_input_byte").str())) &&
        succeeded(verifyValue(rule.nsPerOutputByte,
                              (prefix + ".ns_per_output_byte").str())));
  };
  if (failed(verifyRule(digitalFallback, "digital_fallback")))
    return failure();
  for (const auto &entry : digitalTaskKinds) {
    if (entry.first().empty())
      return anchor->emitError("mapping cost task kind must not be empty");
    if (failed(
            verifyRule(entry.second,
                       (Twine("digital_task_kinds.") + entry.first()).str())))
      return failure();
  }
  if (failed(verifyValue(analog.loadFixedNs, "analog.load_fixed_ns")) ||
      failed(verifyValue(analog.loadNsPerByte, "analog.load_ns_per_byte")) ||
      failed(verifyValue(analog.executeNs, "analog.execute_ns")) ||
      failed(verifyValue(analog.storeFixedNs, "analog.store_fixed_ns")) ||
      failed(verifyValue(analog.storeNsPerByte, "analog.store_ns_per_byte")) ||
      failed(verifyValue(runtime.taskDispatchNs, "runtime.task_dispatch_ns")) ||
      failed(verifyValue(runtime.routeSetupNs, "runtime.route_setup_ns")) ||
      failed(verifyValue(network.hopPipelineNs, "network.hop_pipeline_ns")) ||
      failed(verifyValue(network.injectFixedNs, "network.inject_fixed_ns")) ||
      failed(verifyValue(network.ejectFixedNs, "network.eject_fixed_ns")) ||
      failed(verifyValue(network.dmaNsPerByte, "network.dma_ns_per_byte")))
    return failure();
  return success();
}

MappingCostProfile
getLegacyMappingCostProfile(const MappingHardwareModel &hardware) {
  MappingCostProfile profile;
  profile.clockFrequencyHz = hardware.clockFrequencyHz;
  profile.useLegacyFormula = true;
  int64_t vectorWidth = hardware.digitalVectorBitsPerCycle / 32;
  int64_t effectiveOps = std::max(hardware.digitalIssueWidth, vectorWidth);
  profile.digitalFallback.nsPerWorkItem =
      1.0e9 / static_cast<double>(hardware.clockFrequencyHz) /
      static_cast<double>(effectiveOps);
  profile.analog.executeNs = hardware.analogMVMLatencyNs;
  profile.analog.loadNsPerByte =
      8.0e9 / static_cast<double>(hardware.analogIOBitsPerCycle) /
      static_cast<double>(hardware.clockFrequencyHz);
  profile.analog.storeNsPerByte = profile.analog.loadNsPerByte;
  profile.network.wordBits = hardware.networkWordBits;
  profile.network.hopPipelineNs =
      static_cast<double>(hardware.networkHopCycles) * 1.0e9 /
      static_cast<double>(hardware.clockFrequencyHz);

  std::string fingerprint =
      (Twine(profile.name) + ":" + Twine(hardware.clockFrequencyHz) + ":" +
       Twine(hardware.analogMVMLatencyNs) + ":" +
       Twine(hardware.analogIOBitsPerCycle) + ":" +
       Twine(hardware.digitalIssueWidth) + ":" +
       Twine(hardware.digitalVectorBitsPerCycle) + ":" +
       Twine(hardware.networkWordBits) + ":" + Twine(hardware.networkHopCycles))
          .str();
  llvm::SHA256 sha;
  sha.update(fingerprint);
  profile.contentHash = digestToHex(sha.final());
  return profile;
}

FailureOr<MappingCostProfile>
loadMappingCostProfile(StringRef path, const MappingHardwareModel &hardware,
                       Operation *anchor) {
  if (path.empty())
    return getLegacyMappingCostProfile(hardware);

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    anchor->emitError("cannot read mapping cost profile '")
        << path << "': " << buffer.getError().message();
    return failure();
  }
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    anchor->emitError("cannot parse mapping cost profile '")
        << path << "': " << llvm::toString(parsed.takeError());
    return failure();
  }
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    anchor->emitError("mapping cost profile root must be a JSON object");
    return failure();
  }
  if (failed(rejectUnknownKeys(*root,
                               {"schema_version", "name", "source",
                                "clock_frequency_hz", "digital_fallback",
                                "digital_task_kinds", "analog", "runtime",
                                "network"},
                               anchor, "mapping cost profile")))
    return failure();

  MappingCostProfile profile;
  profile.useLegacyFormula = false;
  profile.clockFrequencyHz = hardware.clockFrequencyHz;
  profile.network.wordBits = hardware.networkWordBits;
  profile.network.hopPipelineNs =
      static_cast<double>(hardware.networkHopCycles) * 1.0e9 /
      static_cast<double>(hardware.clockFrequencyHz);

  std::optional<int64_t> schema = root->getInteger("schema_version");
  std::optional<StringRef> name = root->getString("name");
  std::optional<StringRef> source = root->getString("source");
  if (!schema || !name || !source) {
    anchor->emitError(
        "mapping cost profile requires schema_version, name, and source");
    return failure();
  }
  profile.schemaVersion = *schema;
  profile.name = name->str();
  profile.source = source->str();
  FailureOr<int64_t> clock = readPositiveInteger(
      *root, "clock_frequency_hz", hardware.clockFrequencyHz, anchor,
      "mapping cost profile");
  if (failed(clock))
    return failure();
  profile.clockFrequencyHz = *clock;

  if (const llvm::json::Object *fallback =
          root->getObject("digital_fallback")) {
    FailureOr<TaskCostRule> rule =
        parseTaskCostRule(*fallback, anchor, "digital_fallback");
    if (failed(rule))
      return failure();
    profile.digitalFallback = *rule;
  } else if (root->get("digital_fallback")) {
    anchor->emitError("digital_fallback must be a JSON object");
    return failure();
  }

  if (const llvm::json::Object *kinds = root->getObject("digital_task_kinds")) {
    for (const auto &entry : *kinds) {
      const llvm::json::Object *ruleObject = entry.second.getAsObject();
      if (!ruleObject) {
        anchor->emitError("digital task-kind rule '")
            << StringRef(entry.first) << "' must be a JSON object";
        return failure();
      }
      FailureOr<TaskCostRule> rule = parseTaskCostRule(
          *ruleObject, anchor,
          (Twine("digital_task_kinds.") + StringRef(entry.first)).str());
      if (failed(rule))
        return failure();
      profile.digitalTaskKinds[StringRef(entry.first)] = *rule;
    }
  } else if (root->get("digital_task_kinds")) {
    anchor->emitError("digital_task_kinds must be a JSON object");
    return failure();
  }

  if (const llvm::json::Object *analog = root->getObject("analog")) {
    if (failed(rejectUnknownKeys(*analog,
                                 {"load_fixed_ns", "load_ns_per_byte",
                                  "execute_ns", "store_fixed_ns",
                                  "store_ns_per_byte"},
                                 anchor, "analog")))
      return failure();
    FailureOr<double> loadFixed =
        readNumber(*analog, "load_fixed_ns", 0.0, anchor, "analog");
    FailureOr<double> loadByte =
        readNumber(*analog, "load_ns_per_byte", 0.0, anchor, "analog");
    FailureOr<double> execute =
        readNumber(*analog, "execute_ns", 0.0, anchor, "analog");
    FailureOr<double> storeFixed =
        readNumber(*analog, "store_fixed_ns", 0.0, anchor, "analog");
    FailureOr<double> storeByte =
        readNumber(*analog, "store_ns_per_byte", 0.0, anchor, "analog");
    if (failed(loadFixed) || failed(loadByte) || failed(execute) ||
        failed(storeFixed) || failed(storeByte))
      return failure();
    profile.analog = {*loadFixed, *loadByte, *execute, *storeFixed, *storeByte};
  } else if (root->get("analog")) {
    anchor->emitError("analog must be a JSON object");
    return failure();
  }

  if (const llvm::json::Object *runtime = root->getObject("runtime")) {
    if (failed(rejectUnknownKeys(*runtime,
                                 {"task_dispatch_ns", "route_setup_ns"}, anchor,
                                 "runtime")))
      return failure();
    FailureOr<double> dispatch =
        readNumber(*runtime, "task_dispatch_ns", 0.0, anchor, "runtime");
    FailureOr<double> route =
        readNumber(*runtime, "route_setup_ns", 0.0, anchor, "runtime");
    if (failed(dispatch) || failed(route))
      return failure();
    profile.runtime = {*dispatch, *route};
  } else if (root->get("runtime")) {
    anchor->emitError("runtime must be a JSON object");
    return failure();
  }

  if (const llvm::json::Object *network = root->getObject("network")) {
    if (failed(rejectUnknownKeys(*network,
                                 {"word_bits", "hop_pipeline_ns",
                                  "inject_fixed_ns", "eject_fixed_ns",
                                  "dma_ns_per_byte"},
                                 anchor, "network")))
      return failure();
    FailureOr<int64_t> wordBits = readPositiveInteger(
        *network, "word_bits", hardware.networkWordBits, anchor, "network");
    FailureOr<double> hop =
        readNumber(*network, "hop_pipeline_ns", profile.network.hopPipelineNs,
                   anchor, "network");
    FailureOr<double> inject =
        readNumber(*network, "inject_fixed_ns", 0.0, anchor, "network");
    FailureOr<double> eject =
        readNumber(*network, "eject_fixed_ns", 0.0, anchor, "network");
    FailureOr<double> dma =
        readNumber(*network, "dma_ns_per_byte", 0.0, anchor, "network");
    if (failed(wordBits) || failed(hop) || failed(inject) || failed(eject) ||
        failed(dma))
      return failure();
    profile.network = {*wordBits, *hop, *inject, *eject, *dma};
  } else if (root->get("network")) {
    anchor->emitError("network must be a JSON object");
    return failure();
  }

  llvm::SHA256 sha;
  sha.update(buffer.get()->getBuffer());
  profile.contentHash = digestToHex(sha.final());
  if (failed(profile.verify(hardware, anchor)))
    return failure();
  return profile;
}

DictionaryAttr serializeMappingCostProfile(MLIRContext *context,
                                           const MappingCostProfile &profile) {
  Builder builder(context);
  SmallVector<std::string> kinds;
  kinds.reserve(profile.digitalTaskKinds.size());
  for (const auto &entry : profile.digitalTaskKinds)
    kinds.push_back(entry.first().str());
  llvm::sort(kinds);

  SmallVector<Attribute> taskRules;
  taskRules.reserve(kinds.size());
  for (const std::string &kind : kinds)
    taskRules.push_back(serializeTaskRule(
        context, kind, profile.digitalTaskKinds.lookup(kind)));

  auto dictionary = [&](ArrayRef<NamedAttribute> values) {
    return DictionaryAttr::get(context, values);
  };
  DictionaryAttr analog = dictionary({
      builder.getNamedAttr("load_fixed_ns",
                           builder.getF64FloatAttr(profile.analog.loadFixedNs)),
      builder.getNamedAttr(
          "load_ns_per_byte",
          builder.getF64FloatAttr(profile.analog.loadNsPerByte)),
      builder.getNamedAttr("execute_ns",
                           builder.getF64FloatAttr(profile.analog.executeNs)),
      builder.getNamedAttr("store_fixed_ns", builder.getF64FloatAttr(
                                                 profile.analog.storeFixedNs)),
      builder.getNamedAttr(
          "store_ns_per_byte",
          builder.getF64FloatAttr(profile.analog.storeNsPerByte)),
  });
  DictionaryAttr runtime = dictionary({
      builder.getNamedAttr(
          "task_dispatch_ns",
          builder.getF64FloatAttr(profile.runtime.taskDispatchNs)),
      builder.getNamedAttr("route_setup_ns", builder.getF64FloatAttr(
                                                 profile.runtime.routeSetupNs)),
  });
  DictionaryAttr network = dictionary({
      builder.getNamedAttr("word_bits",
                           builder.getI64IntegerAttr(profile.network.wordBits)),
      builder.getNamedAttr(
          "hop_pipeline_ns",
          builder.getF64FloatAttr(profile.network.hopPipelineNs)),
      builder.getNamedAttr(
          "inject_fixed_ns",
          builder.getF64FloatAttr(profile.network.injectFixedNs)),
      builder.getNamedAttr("eject_fixed_ns", builder.getF64FloatAttr(
                                                 profile.network.ejectFixedNs)),
      builder.getNamedAttr(
          "dma_ns_per_byte",
          builder.getF64FloatAttr(profile.network.dmaNsPerByte)),
  });

  return dictionary({
      builder.getNamedAttr("schema_version",
                           builder.getI64IntegerAttr(profile.schemaVersion)),
      builder.getNamedAttr("name", builder.getStringAttr(profile.name)),
      builder.getNamedAttr("source", builder.getStringAttr(profile.source)),
      builder.getNamedAttr("content_hash",
                           builder.getStringAttr(profile.contentHash)),
      builder.getNamedAttr("clock_frequency_hz",
                           builder.getI64IntegerAttr(profile.clockFrequencyHz)),
      builder.getNamedAttr("use_legacy_formula",
                           builder.getBoolAttr(profile.useLegacyFormula)),
      builder.getNamedAttr(
          "digital_fallback",
          serializeTaskRule(context, "", profile.digitalFallback)),
      builder.getNamedAttr("digital_task_kinds",
                           builder.getArrayAttr(taskRules)),
      builder.getNamedAttr("analog", analog),
      builder.getNamedAttr("runtime", runtime),
      builder.getNamedAttr("network", network),
  });
}

FailureOr<MappingCostProfile>
deserializeMappingCostProfile(DictionaryAttr attr,
                              const MappingHardwareModel &hardware,
                              Operation *anchor) {
  MappingCostProfile profile;
  auto schema = attr.getAs<IntegerAttr>("schema_version");
  auto name = attr.getAs<StringAttr>("name");
  auto source = attr.getAs<StringAttr>("source");
  auto hash = attr.getAs<StringAttr>("content_hash");
  auto legacy = attr.getAs<BoolAttr>("use_legacy_formula");
  auto fallback = attr.getAs<DictionaryAttr>("digital_fallback");
  auto taskKinds = attr.getAs<ArrayAttr>("digital_task_kinds");
  auto analog = attr.getAs<DictionaryAttr>("analog");
  auto runtime = attr.getAs<DictionaryAttr>("runtime");
  auto network = attr.getAs<DictionaryAttr>("network");
  FailureOr<int64_t> clock = getPositiveI64(attr, "clock_frequency_hz", anchor);
  if (!schema || !name || !source || !hash || !legacy || !fallback ||
      !taskKinds || !analog || !runtime || !network || failed(clock)) {
    anchor->emitError("malformed serialized mapping cost profile");
    return failure();
  }
  profile.schemaVersion = schema.getInt();
  profile.name = name.getValue().str();
  profile.source = source.getValue().str();
  profile.contentHash = hash.getValue().str();
  profile.clockFrequencyHz = *clock;
  profile.useLegacyFormula = legacy.getValue();

  FailureOr<TaskCostRule> fallbackRule = deserializeTaskRule(fallback, anchor);
  if (failed(fallbackRule))
    return failure();
  profile.digitalFallback = *fallbackRule;
  for (Attribute value : taskKinds) {
    auto ruleAttr = dyn_cast<DictionaryAttr>(value);
    auto kind = ruleAttr ? ruleAttr.getAs<StringAttr>("kind") : StringAttr();
    if (!ruleAttr || !kind || kind.getValue().empty()) {
      anchor->emitError("malformed serialized digital task-kind rule");
      return failure();
    }
    FailureOr<TaskCostRule> rule = deserializeTaskRule(ruleAttr, anchor);
    if (failed(rule))
      return failure();
    if (!profile.digitalTaskKinds.try_emplace(kind.getValue(), *rule).second) {
      anchor->emitError("duplicate serialized digital task-kind rule '")
          << kind.getValue() << "'";
      return failure();
    }
  }

  FailureOr<double> loadFixed = getF64(analog, "load_fixed_ns", anchor);
  FailureOr<double> loadByte = getF64(analog, "load_ns_per_byte", anchor);
  FailureOr<double> execute = getF64(analog, "execute_ns", anchor);
  FailureOr<double> storeFixed = getF64(analog, "store_fixed_ns", anchor);
  FailureOr<double> storeByte = getF64(analog, "store_ns_per_byte", anchor);
  FailureOr<double> dispatch = getF64(runtime, "task_dispatch_ns", anchor);
  FailureOr<double> route = getF64(runtime, "route_setup_ns", anchor);
  FailureOr<int64_t> wordBits = getPositiveI64(network, "word_bits", anchor);
  FailureOr<double> hop = getF64(network, "hop_pipeline_ns", anchor);
  FailureOr<double> inject = getF64(network, "inject_fixed_ns", anchor);
  FailureOr<double> eject = getF64(network, "eject_fixed_ns", anchor);
  FailureOr<double> dma = getF64(network, "dma_ns_per_byte", anchor);
  if (failed(loadFixed) || failed(loadByte) || failed(execute) ||
      failed(storeFixed) || failed(storeByte) || failed(dispatch) ||
      failed(route) || failed(wordBits) || failed(hop) || failed(inject) ||
      failed(eject) || failed(dma))
    return failure();
  profile.analog = {*loadFixed, *loadByte, *execute, *storeFixed, *storeByte};
  profile.runtime = {*dispatch, *route};
  profile.network = {*wordBits, *hop, *inject, *eject, *dma};
  if (failed(profile.verify(hardware, anchor)))
    return failure();
  return profile;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
