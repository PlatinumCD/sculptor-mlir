#include "RecursiveForkJoinPlanner.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>

namespace {

using namespace mlir;
using namespace mlir::sculptor;
using namespace mlir::sculptor::mapping;

enum class ComputeAtomKind {
  Operation,
  MVMBody,
};

struct ComputeAtom {
  int64_t id = -1;
  ComputeAtomKind kind = ComputeAtomKind::Operation;
  int64_t operationId = -1;
  int64_t mvmWaveId = -1;
  SmallVector<int64_t> memberOperationIds;
};

enum class ComputeRegionKind {
  Atom,
  Sequence,
  Parallel,
};

struct ComputeRegion {
  int64_t id = -1;
  ComputeRegionKind kind = ComputeRegionKind::Atom;
  int64_t atomId = -1;
  SmallVector<int64_t> childIds;
};

struct ComputeRegionTree {
  int64_t rootId = -1;
  SmallVector<ComputeAtom, 0> atoms;
  SmallVector<ComputeRegion, 0> regions;
  bool usedConservativeFallback = false;
};

struct RegionParseResult {
  bool valid = true;
  std::optional<int64_t> rootId;
};

class RecursiveRegionBuilder {
public:
  RecursiveRegionBuilder(const ComputeGraph &graph,
                         ArrayRef<int64_t> includedOperationIds,
                         Operation *anchor,
                         bool exposeParallelMVMCohorts = true)
      : graph(graph), includedOperationIds(includedOperationIds),
        anchor(anchor),
        exposeParallelMVMCohorts(exposeParallelMVMCohorts) {}

  FailureOr<ComputeRegionTree> build() {
    if (includedOperationIds.empty()) {
      anchor->emitError(
          "recursive-fork-join requires at least one compute operation");
      return failure();
    }
    if (failed(buildDAG()) || failed(computePostDominators()))
      return failure();

    RegionParseResult parsed = parsePath(virtualSourceId, virtualSinkId);
    if (!parsed.valid || !parsed.rootId ||
        emittedAtoms.size() != tree.atoms.size()) {
      resetRegions();
      FailureOr<int64_t> fallback = buildConservativeSequence();
      if (failed(fallback))
        return failure();
      tree.rootId = *fallback;
      tree.usedConservativeFallback = true;
    } else {
      tree.rootId = *parsed.rootId;
    }

    if (exposeParallelMVMCohorts)
      formParallelMVMWaveCohorts(tree.rootId);
    serializeSharedLaneBindingGroups(tree.rootId);
    return std::move(tree);
  }

private:
  SmallVector<int64_t> getWaveOperationIds(const MVMWave &wave) const {
    SmallVector<int64_t> operationIds;
    operationIds.append(wave.vectorTileOperationIds.begin(),
                        wave.vectorTileOperationIds.end());
    operationIds.append(wave.physicalMVMOperationIds.begin(),
                        wave.physicalMVMOperationIds.end());
    if (wave.recombineOperationId)
      operationIds.push_back(*wave.recombineOperationId);
    if (wave.biasAddOperationId)
      operationIds.push_back(*wave.biasAddOperationId);
    return operationIds;
  }

  LogicalResult buildDAG() {
    int64_t operationCount = static_cast<int64_t>(graph.operations.size());
    includedOperations.assign(operationCount, false);
    operationToAtom.assign(operationCount, -1);
    SmallVector<int64_t> graphTopologicalRank(operationCount, -1);

    DenseSet<int64_t> requested;
    for (int64_t operationId : includedOperationIds) {
      if (operationId < 0 || operationId >= operationCount ||
          !requested.insert(operationId).second) {
        anchor->emitError(
            "recursive-fork-join received an invalid compute operation ID");
        return failure();
      }
      includedOperations[operationId] = true;
    }

    SmallVector<int64_t> orderedOperations;
    orderedOperations.reserve(includedOperationIds.size());
    for (int64_t operationId : graph.topologicalOrder) {
      if (!includedOperations[operationId])
        continue;
      graphTopologicalRank[operationId] =
          static_cast<int64_t>(orderedOperations.size());
      orderedOperations.push_back(operationId);
    }
    if (orderedOperations.size() != includedOperationIds.size()) {
      anchor->emitError(
          "recursive-fork-join cannot find every operation in topological "
          "order");
      return failure();
    }

    SmallVector<int64_t> operationToWave(operationCount, -1);
    DenseMap<int64_t, SmallVector<int64_t>> waveMembers;
    {
      DenseSet<int64_t> knownWaveIds;
      for (const MVMWave &wave : graph.mvmWaves) {
        SmallVector<int64_t> members = getWaveOperationIds(wave);
        if (wave.id < 0 || !knownWaveIds.insert(wave.id).second ||
            members.empty()) {
          anchor->emitError(
              "recursive-fork-join found malformed MVM-wave identity");
          return failure();
        }
        DenseSet<int64_t> uniqueMembers;
        int64_t includedMemberCount = 0;
        for (int64_t operationId : members) {
          if (operationId < 0 || operationId >= operationCount ||
              !uniqueMembers.insert(operationId).second) {
            anchor->emitError(
                "recursive-fork-join found malformed MVM-wave metadata");
            return failure();
          }
          includedMemberCount += includedOperations[operationId] ? 1 : 0;
        }
        if (includedMemberCount == 0)
          continue;
        if (includedMemberCount != static_cast<int64_t>(members.size())) {
          anchor->emitError(
              "recursive-fork-join cannot collapse only part of an MVM "
              "body");
          return failure();
        }
        llvm::sort(members, [&](int64_t left, int64_t right) {
          return std::pair(graphTopologicalRank[left], left) <
                 std::pair(graphTopologicalRank[right], right);
        });
        for (int64_t operationId : members) {
          if (operationToWave[operationId] >= 0) {
            graph.operations[operationId].operation->emitError(
                "operation belongs to multiple MVM bodies");
            return failure();
          }
          operationToWave[operationId] = wave.id;
        }
        waveMembers[wave.id] = std::move(members);
      }
    }

    DenseSet<int64_t> emittedWaves;
    for (int64_t operationId : orderedOperations) {
      int64_t waveId = operationToWave[operationId];
      if (waveId >= 0) {
        if (!emittedWaves.insert(waveId).second)
          continue;
        ComputeAtom atom;
        atom.id = static_cast<int64_t>(tree.atoms.size());
        atom.kind = ComputeAtomKind::MVMBody;
        atom.mvmWaveId = waveId;
        atom.memberOperationIds = waveMembers.lookup(waveId);
        for (int64_t memberId : atom.memberOperationIds)
          operationToAtom[memberId] = atom.id;
        tree.atoms.push_back(std::move(atom));
        continue;
      }

      ComputeAtom atom;
      atom.id = static_cast<int64_t>(tree.atoms.size());
      atom.kind = ComputeAtomKind::Operation;
      atom.operationId = operationId;
      atom.memberOperationIds.push_back(operationId);
      operationToAtom[operationId] = atom.id;
      tree.atoms.push_back(std::move(atom));
    }
    for (int64_t operationId : orderedOperations) {
      if (operationToAtom[operationId] < 0) {
        graph.operations[operationId].operation->emitError(
            "recursive-fork-join did not assign the operation to an atom");
        return failure();
      }
    }

    int64_t atomCount = static_cast<int64_t>(tree.atoms.size());
    virtualSourceId = atomCount;
    virtualSinkId = atomCount + 1;
    int64_t nodeCount = atomCount + 2;
    successors.resize(nodeCount);
    predecessors.resize(nodeCount);
    topologicalRank.assign(nodeCount, -1);

    auto addEdge = [&](int64_t source, int64_t target) {
      if (source == target || llvm::is_contained(successors[source], target))
        return;
      successors[source].push_back(target);
      predecessors[target].push_back(source);
    };
    for (const ComputeTensor &tensor : graph.tensors) {
      for (int64_t producerId : tensor.producerOperations) {
        if (producerId < 0 || producerId >= operationCount ||
            !includedOperations[producerId])
          continue;
        for (int64_t consumerId : tensor.consumerOperations) {
          if (consumerId < 0 || consumerId >= operationCount ||
              !includedOperations[consumerId])
            continue;
          addEdge(operationToAtom[producerId], operationToAtom[consumerId]);
        }
      }
    }

    SmallVector<int64_t> indegree(atomCount);
    std::set<std::pair<int64_t, int64_t>> ready;
    for (const ComputeAtom &atom : tree.atoms) {
      indegree[atom.id] = static_cast<int64_t>(predecessors[atom.id].size());
      if (indegree[atom.id] == 0) {
        int64_t rank = graphTopologicalRank[atom.memberOperationIds.front()];
        ready.insert({rank, atom.id});
      }
    }
    while (!ready.empty()) {
      int64_t atomId = ready.begin()->second;
      ready.erase(ready.begin());
      atomTopologicalOrder.push_back(atomId);
      for (int64_t successor : successors[atomId]) {
        if (--indegree[successor] != 0)
          continue;
        int64_t rank = graphTopologicalRank[
            tree.atoms[successor].memberOperationIds.front()];
        ready.insert({rank, successor});
      }
    }
    if (atomTopologicalOrder.size() != tree.atoms.size()) {
      anchor->emitError(
          "recursive-fork-join cannot collapse a non-convex MVM body");
      return failure();
    }

    for (int64_t atomId : atomTopologicalOrder) {
      if (predecessors[atomId].empty())
        addEdge(virtualSourceId, atomId);
      if (successors[atomId].empty())
        addEdge(atomId, virtualSinkId);
    }

    topologicalOrder.push_back(virtualSourceId);
    topologicalOrder.append(atomTopologicalOrder.begin(),
                            atomTopologicalOrder.end());
    topologicalOrder.push_back(virtualSinkId);
    for (auto [rank, nodeId] : llvm::enumerate(topologicalOrder))
      topologicalRank[nodeId] = static_cast<int64_t>(rank);

    for (int64_t source = 0; source < nodeCount; ++source) {
      llvm::sort(successors[source], [&](int64_t left, int64_t right) {
        return std::pair(topologicalRank[left], left) <
               std::pair(topologicalRank[right], right);
      });
      for (int64_t target : successors[source]) {
        if (topologicalRank[source] >= topologicalRank[target]) {
          anchor->emitError(
              "recursive-fork-join requires an acyclic compute graph");
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult computePostDominators() {
    int64_t nodeCount = static_cast<int64_t>(successors.size());
    postDominators.assign(nodeCount, llvm::BitVector(nodeCount));
    immediatePostDominator.assign(nodeCount, -1);
    postDominators[virtualSinkId].set(virtualSinkId);

    for (auto iterator = std::next(topologicalOrder.rbegin());
         iterator != topologicalOrder.rend(); ++iterator) {
      int64_t nodeId = *iterator;
      if (successors[nodeId].empty()) {
        anchor->emitError(
            "recursive-fork-join found a non-sink node without a successor");
        return failure();
      }

      llvm::BitVector intersection = postDominators[successors[nodeId].front()];
      for (int64_t successor : llvm::drop_begin(successors[nodeId]))
        intersection &= postDominators[successor];
      intersection.set(nodeId);
      postDominators[nodeId] = std::move(intersection);
    }

    for (int64_t nodeId : topologicalOrder) {
      if (nodeId == virtualSinkId)
        continue;
      int64_t nearest = -1;
      for (int64_t candidate = postDominators[nodeId].find_first();
           candidate >= 0;
           candidate = postDominators[nodeId].find_next(candidate)) {
        if (candidate == nodeId)
          continue;
        if (nearest < 0 ||
            topologicalRank[candidate] < topologicalRank[nearest])
          nearest = candidate;
      }
      if (nearest < 0) {
        anchor->emitError(
            "recursive-fork-join cannot determine an immediate "
            "post-dominator");
        return failure();
      }
      immediatePostDominator[nodeId] = nearest;
    }
    return success();
  }

  int64_t addAtom(int64_t atomId) {
    int64_t id = static_cast<int64_t>(tree.regions.size());
    ComputeRegion region;
    region.id = id;
    region.kind = ComputeRegionKind::Atom;
    region.atomId = atomId;
    tree.regions.push_back(std::move(region));
    return id;
  }

  std::optional<int64_t> addCompound(ComputeRegionKind kind,
                                     ArrayRef<int64_t> children) {
    if (children.empty())
      return std::nullopt;
    if (children.size() == 1)
      return children.front();

    int64_t id = static_cast<int64_t>(tree.regions.size());
    ComputeRegion region;
    region.id = id;
    region.kind = kind;
    region.childIds.assign(children.begin(), children.end());
    tree.regions.push_back(std::move(region));
    return id;
  }

  bool collectUntil(int64_t startId, int64_t stopId,
                    DenseSet<int64_t> &members) const {
    SmallVector<int64_t> worklist{startId};
    while (!worklist.empty()) {
      int64_t nodeId = worklist.pop_back_val();
      if (nodeId == stopId)
        continue;
      if (nodeId == virtualSinkId || nodeId == virtualSourceId ||
          nodeId < 0 ||
          nodeId >= static_cast<int64_t>(tree.atoms.size()))
        return false;
      if (!members.insert(nodeId).second)
        continue;
      for (int64_t successor : successors[nodeId])
        worklist.push_back(successor);
    }
    return true;
  }

  RegionParseResult buildFlatRegion(ArrayRef<int64_t> atomIds) {
    SmallVector<int64_t> ordered(atomIds.begin(), atomIds.end());
    llvm::sort(ordered, [&](int64_t left, int64_t right) {
      return std::pair(topologicalRank[left], left) <
             std::pair(topologicalRank[right], right);
    });
    SmallVector<int64_t> children;
    for (int64_t atomId : ordered) {
      if (!emittedAtoms.insert(atomId).second)
        return {/*valid=*/false, std::nullopt};
      children.push_back(addAtom(atomId));
    }
    return {/*valid=*/true,
            addCompound(ComputeRegionKind::Sequence, children)};
  }

  bool isMVMBodyRegion(int64_t regionId, int64_t &atomId) const {
    if (regionId < 0 || regionId >= static_cast<int64_t>(tree.regions.size()))
      return false;
    const ComputeRegion &region = tree.regions[regionId];
    if (region.kind != ComputeRegionKind::Atom || region.atomId < 0 ||
        region.atomId >= static_cast<int64_t>(tree.atoms.size()) ||
        tree.atoms[region.atomId].kind != ComputeAtomKind::MVMBody)
      return false;
    atomId = region.atomId;
    return true;
  }

  static bool containsSameNodes(ArrayRef<int64_t> left,
                                ArrayRef<int64_t> right) {
    if (left.size() != right.size())
      return false;
    return llvm::all_of(left, [&](int64_t nodeId) {
      return llvm::is_contained(right, nodeId);
    });
  }

  bool haveMatchingPredecessorFrontiers(int64_t leftAtomId,
                                        int64_t rightAtomId) const {
    return containsSameNodes(predecessors[leftAtomId],
                             predecessors[rightAtomId]);
  }

  bool reaches(int64_t sourceId, int64_t targetId) const {
    if (sourceId == targetId)
      return true;
    llvm::BitVector visited(successors.size());
    SmallVector<int64_t> worklist{sourceId};
    visited.set(sourceId);
    while (!worklist.empty()) {
      int64_t currentId = worklist.pop_back_val();
      for (int64_t successorId : successors[currentId]) {
        if (successorId == targetId)
          return true;
        if (visited.test(successorId))
          continue;
        visited.set(successorId);
        worklist.push_back(successorId);
      }
    }
    return false;
  }

  bool canJoinMVMWaveCohort(int64_t candidateAtomId,
                            ArrayRef<int64_t> cohortAtomIds) const {
    if (cohortAtomIds.empty() ||
        !haveMatchingPredecessorFrontiers(cohortAtomIds.front(),
                                          candidateAtomId))
      return false;
    return llvm::all_of(cohortAtomIds, [&](int64_t memberAtomId) {
      return !reaches(memberAtomId, candidateAtomId) &&
             !reaches(candidateAtomId, memberAtomId);
    });
  }

  // Recover simultaneously ready MVM bodies when intermediate joins, such as
  // balanced reductions, force the general fork/join parser to retain a
  // conservative sequence. Successor frontiers may differ: the sequence after
  // this cohort remains ordered, while lane-binding legalization below
  // serializes bodies that genuinely share one programmed array.
  void formParallelMVMWaveCohorts(int64_t regionId) {
    if (regionId < 0 || regionId >= static_cast<int64_t>(tree.regions.size()))
      return;

    SmallVector<int64_t> originalChildren = tree.regions[regionId].childIds;
    for (int64_t childId : originalChildren)
      formParallelMVMWaveCohorts(childId);

    if (tree.regions[regionId].kind != ComputeRegionKind::Sequence)
      return;

    SmallVector<int64_t> groupedChildren;
    for (size_t index = 0; index < originalChildren.size();) {
      int64_t firstAtomId = -1;
      if (!isMVMBodyRegion(originalChildren[index], firstAtomId)) {
        groupedChildren.push_back(originalChildren[index++]);
        continue;
      }

      SmallVector<int64_t> cohortRegions{originalChildren[index]};
      SmallVector<int64_t> cohortAtoms{firstAtomId};
      size_t nextIndex = index + 1;
      while (nextIndex < originalChildren.size()) {
        int64_t candidateAtomId = -1;
        if (!isMVMBodyRegion(originalChildren[nextIndex], candidateAtomId) ||
            !canJoinMVMWaveCohort(candidateAtomId, cohortAtoms))
          break;
        cohortRegions.push_back(originalChildren[nextIndex]);
        cohortAtoms.push_back(candidateAtomId);
        ++nextIndex;
      }

      if (cohortRegions.size() == 1) {
        groupedChildren.push_back(cohortRegions.front());
      } else {
        groupedChildren.push_back(
            *addCompound(ComputeRegionKind::Parallel, cohortRegions));
      }
      index = nextIndex;
    }
    tree.regions[regionId].childIds = std::move(groupedChildren);
  }

  void collectRegionLaneBindingGroups(int64_t regionId,
                                      DenseSet<int64_t> &groups) const {
    assert(regionId >= 0 &&
           regionId < static_cast<int64_t>(tree.regions.size()) &&
           "compute region must exist");
    const ComputeRegion &region = tree.regions[regionId];
    if (region.kind == ComputeRegionKind::Atom) {
      assert(region.atomId >= 0 &&
             region.atomId < static_cast<int64_t>(tree.atoms.size()) &&
             "compute atom must exist");
      for (int64_t operationId : tree.atoms[region.atomId].memberOperationIds) {
        const std::optional<int64_t> &bindingGroup =
            graph.operations[operationId].laneBindingGroup;
        if (bindingGroup)
          groups.insert(*bindingGroup);
      }
      return;
    }

    for (int64_t childId : region.childIds)
      collectRegionLaneBindingGroups(childId, groups);
  }

  // A programmed analog lane may execute several independent consumers, but
  // those consumers cannot occupy different resource lanes without cloning
  // the matrix. Serialize only spatial siblings that share a lane binding and
  // retain spatial parallelism between all disjoint components.
  void serializeSharedLaneBindingGroups(int64_t regionId) {
    assert(regionId >= 0 &&
           regionId < static_cast<int64_t>(tree.regions.size()) &&
           "compute region must exist");

    SmallVector<int64_t> children = tree.regions[regionId].childIds;
    for (int64_t childId : children)
      serializeSharedLaneBindingGroups(childId);

    if (tree.regions[regionId].kind != ComputeRegionKind::Parallel ||
        children.size() < 2)
      return;

    SmallVector<int64_t> parents(children.size());
    for (auto [index, parent] : llvm::enumerate(parents))
      parent = static_cast<int64_t>(index);

    auto findRoot = [&](int64_t index) {
      while (parents[index] != index) {
        parents[index] = parents[parents[index]];
        index = parents[index];
      }
      return index;
    };
    auto unite = [&](int64_t left, int64_t right) {
      left = findRoot(left);
      right = findRoot(right);
      if (left == right)
        return;
      if (left > right)
        std::swap(left, right);
      parents[right] = left;
    };

    DenseMap<int64_t, int64_t> firstChildByBindingGroup;
    for (auto [childIndex, childId] : llvm::enumerate(children)) {
      DenseSet<int64_t> groups;
      collectRegionLaneBindingGroups(childId, groups);
      for (int64_t group : groups) {
        auto [owner, inserted] = firstChildByBindingGroup.try_emplace(
            group, static_cast<int64_t>(childIndex));
        if (!inserted)
          unite(static_cast<int64_t>(childIndex), owner->second);
      }
    }

    DenseMap<int64_t, int64_t> componentIndexByRoot;
    SmallVector<SmallVector<int64_t>> components;
    for (auto [childIndex, childId] : llvm::enumerate(children)) {
      int64_t root = findRoot(static_cast<int64_t>(childIndex));
      auto [component, inserted] = componentIndexByRoot.try_emplace(
          root, static_cast<int64_t>(components.size()));
      if (inserted)
        components.emplace_back();
      components[component->second].push_back(childId);
    }

    if (components.size() == children.size())
      return;
    if (components.size() == 1) {
      tree.regions[regionId].kind = ComputeRegionKind::Sequence;
      tree.regions[regionId].childIds = std::move(components.front());
      return;
    }

    SmallVector<int64_t> legalizedChildren;
    legalizedChildren.reserve(components.size());
    for (const SmallVector<int64_t> &component : components) {
      if (component.size() == 1) {
        legalizedChildren.push_back(component.front());
        continue;
      }
      legalizedChildren.push_back(
          *addCompound(ComputeRegionKind::Sequence, component));
    }
    tree.regions[regionId].childIds = std::move(legalizedChildren);
  }

  RegionParseResult parseFork(int64_t forkId, int64_t outerStopId) {
    int64_t joinId = immediatePostDominator[forkId];
    if (joinId < 0 || joinId == forkId ||
        (joinId != outerStopId &&
         !postDominators[joinId].test(outerStopId)))
      return {/*valid=*/false, std::nullopt};

    SmallVector<int64_t> branchStarts;
    for (int64_t successor : successors[forkId]) {
      if (successor != joinId)
        branchStarts.push_back(successor);
    }
    if (branchStarts.empty())
      return {/*valid=*/true, std::nullopt};

    SmallVector<DenseSet<int64_t>> branchMembers(branchStarts.size());
    DenseSet<int64_t> allMembers;
    bool disjoint = true;
    for (auto [branchIndex, branchStart] : llvm::enumerate(branchStarts)) {
      if (!collectUntil(branchStart, joinId, branchMembers[branchIndex]))
        return {/*valid=*/false, std::nullopt};
      for (int64_t atomId : branchMembers[branchIndex]) {
        if (!allMembers.insert(atomId).second)
          disjoint = false;
      }
    }

    if (!disjoint) {
      SmallVector<int64_t> flattened(allMembers.begin(), allMembers.end());
      return buildFlatRegion(flattened);
    }

    SmallVector<int64_t> branches;
    for (int64_t branchStart : branchStarts) {
      RegionParseResult branch = parsePath(branchStart, joinId);
      if (!branch.valid)
        return branch;
      if (branch.rootId)
        branches.push_back(*branch.rootId);
    }
    return {/*valid=*/true,
            addCompound(ComputeRegionKind::Parallel, branches)};
  }

  RegionParseResult parsePath(int64_t startId, int64_t stopId) {
    SmallVector<int64_t> sequence;
    DenseSet<int64_t> pathNodes;
    int64_t currentId = startId;
    while (currentId != stopId) {
      if (!pathNodes.insert(currentId).second || currentId == virtualSinkId)
        return {/*valid=*/false, std::nullopt};

      if (currentId != virtualSourceId) {
        if (currentId < 0 ||
            currentId >= static_cast<int64_t>(tree.atoms.size()) ||
            !emittedAtoms.insert(currentId).second)
          return {/*valid=*/false, std::nullopt};
        sequence.push_back(addAtom(currentId));
      }

      if (successors[currentId].empty())
        return {/*valid=*/false, std::nullopt};
      if (successors[currentId].size() == 1) {
        currentId = successors[currentId].front();
        continue;
      }

      RegionParseResult fork = parseFork(currentId, stopId);
      if (!fork.valid)
        return fork;
      if (fork.rootId)
        sequence.push_back(*fork.rootId);
      currentId = immediatePostDominator[currentId];
    }
    return {/*valid=*/true,
            addCompound(ComputeRegionKind::Sequence, sequence)};
  }

  FailureOr<int64_t> buildConservativeSequence() {
    SmallVector<int64_t> children;
    children.reserve(atomTopologicalOrder.size());
    for (int64_t atomId : atomTopologicalOrder) {
      if (!emittedAtoms.insert(atomId).second) {
        anchor->emitError(
            "recursive-fork-join fallback contains a duplicate atom");
        return failure();
      }
      children.push_back(addAtom(atomId));
    }
    std::optional<int64_t> root =
        addCompound(ComputeRegionKind::Sequence, children);
    if (!root) {
      anchor->emitError("recursive-fork-join produced an empty fallback");
      return failure();
    }
    return *root;
  }

  void resetRegions() {
    tree.regions.clear();
    emittedAtoms.clear();
  }

  const ComputeGraph &graph;
  ArrayRef<int64_t> includedOperationIds;
  Operation *anchor;
  bool exposeParallelMVMCohorts;
  ComputeRegionTree tree;
  int64_t virtualSourceId = -1;
  int64_t virtualSinkId = -1;
  SmallVector<bool> includedOperations;
  SmallVector<int64_t> operationToAtom;
  SmallVector<SmallVector<int64_t>> successors;
  SmallVector<SmallVector<int64_t>> predecessors;
  SmallVector<int64_t> atomTopologicalOrder;
  SmallVector<int64_t> topologicalOrder;
  SmallVector<int64_t> topologicalRank;
  SmallVector<llvm::BitVector> postDominators;
  SmallVector<int64_t> immediatePostDominator;
  DenseSet<int64_t> emittedAtoms;
};

void collectSubtreeOperations(
    int64_t nodeId,
    const DenseMap<int64_t, const StructuralRATreeNode *> &nodes,
    DenseSet<int64_t> &operationIds) {
  const StructuralRATreeNode *node = nodes.lookup(nodeId);
  assert(node && "verified RA tree must contain every referenced node");
  if (node->kind == RATreeNodeKind::Leaf) {
    operationIds.insert(node->operationId);
    return;
  }
  for (int64_t childId : node->childIds)
    collectSubtreeOperations(childId, nodes, operationIds);
}

struct SetupPartition {
  std::optional<int64_t> setupRootId;
  SmallVector<int64_t> computeOperationIds;
};

FailureOr<SetupPartition>
findSetupPartition(const MappingProblem &problem) {
  DenseSet<int64_t> expectedSetups;
  SetupPartition partition;
  for (int64_t operationId : problem.graph.topologicalOrder) {
    if (problem.graph.operations[operationId].kind ==
        ComputeOperationKind::MatrixSetup)
      expectedSetups.insert(operationId);
    else
      partition.computeOperationIds.push_back(operationId);
  }
  if (expectedSetups.empty())
    return partition;

  DenseMap<int64_t, const StructuralRATreeNode *> nodes;
  for (const StructuralRATreeNode &node : problem.currentTree.nodes)
    nodes[node.id] = &node;
  const StructuralRATreeNode *root = nodes.lookup(problem.currentTree.rootId);
  if (!root || root->kind != RATreeNodeKind::TemporalCut ||
      root->childIds.size() != 2) {
    problem.anchor->emitError(
        "recursive-fork-join must run after setup-first when matrix setup "
        "operations are present");
    return failure();
  }

  DenseSet<int64_t> actualSetups;
  collectSubtreeOperations(root->childIds.front(), nodes, actualSetups);
  if (actualSetups != expectedSetups) {
    problem.anchor->emitError(
        "recursive-fork-join cannot identify the setup-first subtree");
    return failure();
  }
  partition.setupRootId = root->childIds.front();
  return partition;
}

class RATreeAssembler {
public:
  RATreeAssembler(const MappingProblem &problem,
                  const ComputeRegionTree &regions)
      : problem(problem), regions(regions) {
    tree.workUnits = problem.currentTree.workUnits;
    tree.workUnitEdges = problem.currentTree.workUnitEdges;
    for (const StructuralRATreeNode &node : problem.currentTree.nodes) {
      sourceNodes[node.id] = &node;
      if (node.kind == RATreeNodeKind::Leaf)
        leavesByOperation[node.operationId].push_back(&node);
    }
  }

  FailureOr<ResourceAllocationTree>
  build(std::optional<int64_t> setupRootId) {
    FailureOr<int64_t> computeRoot = buildRegion(regions, regions.rootId);
    if (failed(computeRoot))
      return failure();

    int64_t rootId = *computeRoot;
    if (setupRootId) {
      FailureOr<int64_t> setupRoot = cloneSubtree(*setupRootId);
      if (failed(setupRoot))
        return failure();
      rootId = addCut(RATreeNodeKind::TemporalCut, {*setupRoot, *computeRoot});
    }
    tree.rootId = rootId;
    tree.nodes[rootId].parentId = -1;
    return std::move(tree);
  }

private:
  int64_t addLeaf(int64_t operationId, int64_t workUnitId,
                  int64_t workGroupCount) {
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    tree.nodes.push_back({id, RATreeNodeKind::Leaf, -1, {}, operationId,
                          workUnitId, workGroupCount});
    return id;
  }

  int64_t addCut(RATreeNodeKind kind, ArrayRef<int64_t> children,
                 int64_t workGroupCount = 1) {
    assert(children.size() >= 2 && "cuts require at least two children");
    int64_t id = static_cast<int64_t>(tree.nodes.size());
    StructuralRATreeNode node;
    node.id = id;
    node.kind = kind;
    node.childIds.assign(children.begin(), children.end());
    node.workGroupCount = workGroupCount;
    tree.nodes.push_back(std::move(node));
    for (int64_t childId : children)
      tree.nodes[childId].parentId = id;
    return id;
  }

  FailureOr<int64_t> buildOperation(int64_t operationId) {
    auto leaves = leavesByOperation.find(operationId);
    if (leaves == leavesByOperation.end() || leaves->second.empty()) {
      problem.graph.operations[operationId].operation->emitError(
          "recursive-fork-join cannot resolve the operation's RA leaf");
      return failure();
    }
    llvm::sort(leaves->second,
               [](const StructuralRATreeNode *left,
                  const StructuralRATreeNode *right) {
                 return std::pair(left->workUnitId, left->id) <
                        std::pair(right->workUnitId, right->id);
               });
    SmallVector<int64_t> children;
    for (const StructuralRATreeNode *leaf : leaves->second) {
      children.push_back(addLeaf(operationId, leaf->workUnitId,
                                 leaf->workGroupCount));
    }
    return children.size() == 1
               ? children.front()
               : addCut(RATreeNodeKind::SpatialCut, children);
  }

  FailureOr<int64_t> buildPackedMVMBody(const ComputeAtom &atom) {
    const MVMWave *wave = nullptr;
    for (const MVMWave &candidate : problem.graph.mvmWaves) {
      if (candidate.id == atom.mvmWaveId) {
        wave = &candidate;
        break;
      }
    }
    if (!wave || wave->physicalMVMOperationIds.empty()) {
      problem.anchor->emitError(
          "recursive-fork-join cannot resolve a packed MVM body");
      return failure();
    }

    SmallVector<int64_t> phases;
    for (int64_t operationId : wave->vectorTileOperationIds) {
      FailureOr<int64_t> vectorTile = buildOperation(operationId);
      if (failed(vectorTile))
        return failure();
      phases.push_back(*vectorTile);
    }

    SmallVector<int64_t> analogBranches;
    for (int64_t operationId : wave->physicalMVMOperationIds) {
      FailureOr<int64_t> physicalMVM = buildOperation(operationId);
      if (failed(physicalMVM))
        return failure();
      analogBranches.push_back(*physicalMVM);
    }
    phases.push_back(analogBranches.size() == 1
                         ? analogBranches.front()
                         : addCut(RATreeNodeKind::SpatialCut, analogBranches));

    if (wave->recombineOperationId) {
      FailureOr<int64_t> recombine =
          buildOperation(*wave->recombineOperationId);
      if (failed(recombine))
        return failure();
      phases.push_back(*recombine);
    }
    if (wave->biasAddOperationId) {
      FailureOr<int64_t> biasAdd = buildOperation(*wave->biasAddOperationId);
      if (failed(biasAdd))
        return failure();
      phases.push_back(*biasAdd);
    }

    return phases.size() == 1
               ? phases.front()
               : addCut(RATreeNodeKind::TemporalCut, phases);
  }

  FailureOr<int64_t> buildAtom(const ComputeAtom &atom) {
    if (atom.kind == ComputeAtomKind::Operation)
      return buildOperation(atom.operationId);
    return buildPackedMVMBody(atom);
  }

  FailureOr<int64_t> buildRegion(const ComputeRegionTree &regionTree,
                                 int64_t regionId) {
    if (regionId < 0 ||
        regionId >= static_cast<int64_t>(regionTree.regions.size())) {
      problem.anchor->emitError(
          "recursive-fork-join cannot resolve a compute region");
      return failure();
    }
    const ComputeRegion &region = regionTree.regions[regionId];
    if (region.kind == ComputeRegionKind::Atom) {
      if (region.atomId < 0 ||
          region.atomId >= static_cast<int64_t>(regionTree.atoms.size())) {
        problem.anchor->emitError(
            "recursive-fork-join cannot resolve a compute atom");
        return failure();
      }
      return buildAtom(regionTree.atoms[region.atomId]);
    }

    SmallVector<int64_t> children;
    for (int64_t childId : region.childIds) {
      FailureOr<int64_t> child = buildRegion(regionTree, childId);
      if (failed(child))
        return failure();
      children.push_back(*child);
    }
    if (children.empty()) {
      problem.anchor->emitError(
          "recursive-fork-join produced an empty compound region");
      return failure();
    }
    if (children.size() == 1)
      return children.front();
    return addCut(region.kind == ComputeRegionKind::Sequence
                      ? RATreeNodeKind::TemporalCut
                      : RATreeNodeKind::SpatialCut,
                  children);
  }

  FailureOr<int64_t> cloneSubtree(int64_t sourceNodeId) {
    const StructuralRATreeNode *source = sourceNodes.lookup(sourceNodeId);
    if (!source) {
      problem.anchor->emitError(
          "recursive-fork-join cannot clone a missing setup RA node");
      return failure();
    }
    if (source->kind == RATreeNodeKind::Leaf)
      return addLeaf(source->operationId, source->workUnitId,
                     source->workGroupCount);

    SmallVector<int64_t> children;
    for (int64_t childId : source->childIds) {
      FailureOr<int64_t> child = cloneSubtree(childId);
      if (failed(child))
        return failure();
      children.push_back(*child);
    }
    return addCut(source->kind, children, source->workGroupCount);
  }

  const MappingProblem &problem;
  const ComputeRegionTree &regions;
  ResourceAllocationTree tree;
  DenseMap<int64_t, const StructuralRATreeNode *> sourceNodes;
  DenseMap<int64_t, SmallVector<const StructuralRATreeNode *>>
      leavesByOperation;
};

} // namespace

namespace mlir {
namespace sculptor {
namespace mapping {

FailureOr<MappingPlan>
RecursiveForkJoinPlanner::refine(const MappingProblem &problem,
                                 const MappingEvaluator &evaluator) const {
  FailureOr<SetupPartition> partition = findSetupPartition(problem);
  if (failed(partition))
    return failure();

  RecursiveRegionBuilder regionBuilder(
      problem.graph, partition->computeOperationIds, problem.anchor);
  FailureOr<ComputeRegionTree> regions = regionBuilder.build();
  if (failed(regions))
    return failure();

  RATreeAssembler assembler(problem, *regions);
  FailureOr<ResourceAllocationTree> tree =
      assembler.build(partition->setupRootId);
  if (failed(tree))
    return failure();
  FailureOr<ResourceAllocationTree> reindexed =
      reindexResourceAllocationTree(*tree, problem.anchor);
  if (failed(reindexed) || failed(verifyResourceAllocationTree(
                               *reindexed, problem.graph, problem.anchor)))
    return failure();

  FailureOr<MappingEvaluation> evaluation =
      evaluator.evaluate(problem, *reindexed);
  if (failed(evaluation))
    return failure();

  MappingPlan plan;
  plan.plannerName = getName().str();
  plan.objective = problem.objective;
  plan.selectedTree = std::move(*reindexed);
  plan.evaluation = *evaluation;
  plan.candidates.push_back(
      {regions->usedConservativeFallback
           ? "recursive-fork-join-conservative-fallback"
           : "recursive-fork-join",
       *evaluation, /*selected=*/true});
  return plan;
}

} // namespace mapping
} // namespace sculptor
} // namespace mlir
