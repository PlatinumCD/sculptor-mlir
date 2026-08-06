#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "golem/runtime/runtime.h"

namespace {

constexpr size_t kMemRefCopyCallCapacity = 32;
size_t memref_copy_call_count = 0;
std::array<size_t, kMemRefCopyCallCapacity> memref_copy_byte_counts{};

void resetMemRefCopyCalls() {
    memref_copy_call_count = 0;
    memref_copy_byte_counts.fill(0);
}

}  // namespace

extern "C" void mittensMemrefCopyTestHook(size_t byte_count) {
    if (memref_copy_call_count < memref_copy_byte_counts.size()) {
        memref_copy_byte_counts[memref_copy_call_count] = byte_count;
    }
    ++memref_copy_call_count;
}

namespace {

using namespace golem::runtime;

int failures = 0;

struct UnrankedMemRef {
    int64_t rank;
    void* descriptor;
};

extern "C" void memrefCopy(
    int64_t element_size,
    UnrankedMemRef* source,
    UnrankedMemRef* destination
);

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(                                                       \
                stderr,                                                         \
                "%s:%d: check failed: %s\n",                                   \
                __FILE__,                                                       \
                __LINE__,                                                       \
                #condition                                                      \
            );                                                                  \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

TaskStatus executeNoop(
    const Tensor*,
    uint32_t,
    Tensor*,
    uint32_t
) {
    return TaskStatus::Success;
}

uint32_t boot_count = 0;

TaskStatus executeBoot(
    const Tensor*,
    uint32_t input_count,
    Tensor*,
    uint32_t output_count
) {
    if (input_count != 0 || output_count != 0) {
        return TaskStatus::Failure;
    }
    ++boot_count;
    return TaskStatus::Success;
}

TaskStatus executeAddOne(
    const Tensor* inputs,
    uint32_t input_count,
    Tensor* outputs,
    uint32_t output_count
) {
    if (inputs == nullptr ||
        outputs == nullptr ||
        input_count != 1 ||
        output_count != 1 ||
        inputs[0].element_type != ElementType::Float32 ||
        outputs[0].element_type != ElementType::Float32 ||
        inputs[0].rank != 0 ||
        outputs[0].rank != 0 ||
        inputs[0].descriptor == nullptr ||
        outputs[0].descriptor == nullptr) {
        return TaskStatus::Failure;
    }

    *static_cast<float*>(outputs[0].descriptor) =
        *static_cast<const float*>(inputs[0].descriptor) + 1.0F;
    return TaskStatus::Success;
}

struct ScalarMemRefDescriptor {
    void* allocated;
    void* aligned;
    int64_t offset;
};

TaskStatus executeMemRefAddOne(
    const Tensor* inputs,
    uint32_t input_count,
    Tensor* outputs,
    uint32_t output_count
) {
    if (inputs == nullptr ||
        outputs == nullptr ||
        input_count != 1 ||
        output_count != 1 ||
        inputs[0].element_type != ElementType::Float32 ||
        outputs[0].element_type != ElementType::Float32 ||
        inputs[0].rank != 0 ||
        outputs[0].rank != 0) {
        return TaskStatus::Failure;
    }
    const auto* input =
        static_cast<const ScalarMemRefDescriptor*>(inputs[0].descriptor);
    auto* output =
        static_cast<ScalarMemRefDescriptor*>(outputs[0].descriptor);
    if (input == nullptr ||
        output == nullptr ||
        input->aligned == nullptr ||
        output->aligned == nullptr) {
        return TaskStatus::Failure;
    }
    const auto* input_data = static_cast<const float*>(input->aligned);
    auto* output_data = static_cast<float*>(output->aligned);
    output_data[output->offset] =
        input_data[input->offset] + 1.0F;
    return TaskStatus::Success;
}

TaskStatus executeMemRefAddTwo(
    const Tensor* inputs,
    uint32_t input_count,
    Tensor* outputs,
    uint32_t output_count
) {
    if (inputs == nullptr ||
        outputs == nullptr ||
        input_count != 2 ||
        output_count != 1) {
        return TaskStatus::Failure;
    }
    const auto* left =
        static_cast<const ScalarMemRefDescriptor*>(
            inputs[0].descriptor);
    const auto* right =
        static_cast<const ScalarMemRefDescriptor*>(
            inputs[1].descriptor);
    auto* output =
        static_cast<ScalarMemRefDescriptor*>(
            outputs[0].descriptor);
    if (left == nullptr ||
        right == nullptr ||
        output == nullptr ||
        left->aligned == nullptr ||
        right->aligned == nullptr ||
        output->aligned == nullptr) {
        return TaskStatus::Failure;
    }
    *static_cast<float*>(output->aligned) =
        *static_cast<const float*>(left->aligned) +
        *static_cast<const float*>(right->aligned);
    return TaskStatus::Success;
}

void testTensor() {
    float scalar = 1.0F;
    Tensor tensor{ElementType::Float32, 0, &scalar};

    CHECK(elementSizeBytes(ElementType::Int8) == 1);
    CHECK(elementSizeBytes(ElementType::UInt16) == 2);
    CHECK(elementSizeBytes(ElementType::Float32) == 4);
    CHECK(elementSizeBytes(ElementType::Invalid) == 0);
    CHECK(validTensor(tensor));

    tensor.rank = 4;
    CHECK(validTensor(tensor));
    tensor.rank = -1;
    CHECK(!validTensor(tensor));
    tensor.rank = 0;
    tensor.descriptor = nullptr;
    CHECK(!validTensor(tensor));
}

void testScratchpadABI() {
    ScratchpadDMADescriptor descriptor{
        0,
        ScratchpadDMADirection::BackingToScratchpad,
        3,
        UINT32_MAX,
        64,
        128,
        7,
        ScratchpadDMATrigger::Boot,
        UINT32_MAX,
        ScratchpadDMAAsynchronous,
        ScratchpadStorage::Backing,
        ScratchpadStorage::Scratchpad,
        0,
    };
    CHECK(sizeof(descriptor) == 64);
    CHECK(descriptor.valid(256));
    CHECK(!descriptor.valid(128));

    const ScratchpadABI abi{
        ScratchpadDMAFeature,
        256,
        &descriptor,
        1,
    };
    CHECK(abi.valid(256));
    CHECK(!abi.valid(255));

    descriptor.flags = ScratchpadDMAAsynchronous |
                       ScratchpadDMABlocking;
    CHECK(!descriptor.valid(256));
    descriptor.flags = ScratchpadDMAAsynchronous;
    descriptor.destination_storage = ScratchpadStorage::Backing;
    CHECK(!descriptor.valid(256));

    const ScratchpadABI legacy{0, 0, nullptr, 0};
    CHECK(legacy.valid(0));
}

void testMemRefCopy() {
    struct RankedMemRef2D {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[2];
        int64_t strides[2];
    };
    struct RankedMemRef4D {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[4];
        int64_t strides[4];
    };
    struct RankedMemRef3D {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[3];
        int64_t strides[3];
    };

    float source_data[] = {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };
    float destination_data[6]{};
    RankedMemRef2D source{
        source_data,
        source_data,
        0,
        {2, 3},
        {3, 1},
    };
    RankedMemRef2D destination{
        destination_data,
        destination_data,
        0,
        {2, 3},
        {3, 1},
    };
    UnrankedMemRef source_unranked{2, &source};
    UnrankedMemRef destination_unranked{2, &destination};
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &source_unranked,
        &destination_unranked
    );
    for (uint32_t index = 0; index < 6; ++index) {
        CHECK(destination_data[index] == source_data[index]);
    }
    CHECK(memref_copy_call_count == 1);
    CHECK(memref_copy_byte_counts[0] == 6 * sizeof(float));

    alignas(64) float rank4_source_data[24];
    alignas(64) float rank4_destination_data[24]{};
    for (uint32_t index = 0; index < 24; ++index) {
        rank4_source_data[index] = static_cast<float>(index + 1);
    }
    RankedMemRef4D rank4_source{
        rank4_source_data,
        rank4_source_data,
        0,
        {1, 2, 3, 4},
        {24, 12, 4, 1},
    };
    RankedMemRef4D rank4_destination{
        rank4_destination_data,
        rank4_destination_data,
        0,
        {1, 2, 3, 4},
        {24, 12, 4, 1},
    };
    UnrankedMemRef rank4_source_unranked{4, &rank4_source};
    UnrankedMemRef rank4_destination_unranked{
        4,
        &rank4_destination,
    };
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &rank4_source_unranked,
        &rank4_destination_unranked
    );
    for (uint32_t index = 0; index < 24; ++index) {
        CHECK(
            rank4_destination_data[index] ==
            rank4_source_data[index]
        );
    }
    CHECK(memref_copy_call_count == 1);
    CHECK(memref_copy_byte_counts[0] == 24 * sizeof(float));

    float strided_source_data[] = {
        1.0F, 2.0F, 3.0F, -1.0F,
        4.0F, 5.0F, 6.0F, -1.0F,
    };
    float strided_destination_data[8]{};
    RankedMemRef2D strided_source{
        strided_source_data,
        strided_source_data,
        0,
        {2, 3},
        {4, 1},
    };
    RankedMemRef2D strided_destination{
        strided_destination_data,
        strided_destination_data,
        0,
        {2, 3},
        {4, 1},
    };
    UnrankedMemRef strided_source_unranked{2, &strided_source};
    UnrankedMemRef strided_destination_unranked{
        2,
        &strided_destination,
    };
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &strided_source_unranked,
        &strided_destination_unranked
    );
    for (uint32_t index : {0U, 1U, 2U, 4U, 5U, 6U}) {
        CHECK(
            strided_destination_data[index] ==
            strided_source_data[index]
        );
    }
    CHECK(strided_destination_data[3] == 0.0F);
    CHECK(strided_destination_data[7] == 0.0F);
    CHECK(memref_copy_call_count == 2);
    CHECK(memref_copy_byte_counts[0] == 3 * sizeof(float));
    CHECK(memref_copy_byte_counts[1] == 3 * sizeof(float));

    std::array<float, 4 * 2304> qkv_source_data{};
    std::array<float, 4 * 768> qkv_destination_data{};
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 2304; ++column) {
            qkv_source_data[row * 2304 + column] =
                static_cast<float>(row * 2304 + column + 1);
        }
    }
    RankedMemRef3D qkv_source{
        qkv_source_data.data(),
        qkv_source_data.data(),
        0,
        {1, 4, 768},
        {9216, 2304, 1},
    };
    RankedMemRef3D qkv_destination{
        qkv_destination_data.data(),
        qkv_destination_data.data(),
        0,
        {1, 4, 768},
        {3072, 768, 1},
    };
    UnrankedMemRef qkv_source_unranked{3, &qkv_source};
    UnrankedMemRef qkv_destination_unranked{3, &qkv_destination};
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &qkv_source_unranked,
        &qkv_destination_unranked
    );
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 768; ++column) {
            CHECK(
                qkv_destination_data[row * 768 + column] ==
                qkv_source_data[row * 2304 + column]
            );
        }
    }
    CHECK(memref_copy_call_count == 4);
    for (size_t call = 0; call < 4; ++call) {
        CHECK(memref_copy_byte_counts[call] == 768 * sizeof(float));
    }

    float size_one_source_data[8]{};
    float size_one_destination_data[8]{};
    for (size_t outer = 0; outer < 2; ++outer) {
        for (size_t inner = 0; inner < 3; ++inner) {
            size_one_source_data[outer * 5 + inner] =
                static_cast<float>(outer * 10 + inner + 1);
        }
    }
    RankedMemRef3D size_one_source{
        size_one_source_data,
        size_one_source_data,
        0,
        {2, 1, 3},
        {5, 97, 1},
    };
    RankedMemRef3D size_one_destination{
        size_one_destination_data,
        size_one_destination_data,
        0,
        {2, 1, 3},
        {4, 53, 1},
    };
    UnrankedMemRef size_one_source_unranked{3, &size_one_source};
    UnrankedMemRef size_one_destination_unranked{
        3,
        &size_one_destination,
    };
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &size_one_source_unranked,
        &size_one_destination_unranked
    );
    for (size_t outer = 0; outer < 2; ++outer) {
        for (size_t inner = 0; inner < 3; ++inner) {
            CHECK(
                size_one_destination_data[outer * 4 + inner] ==
                size_one_source_data[outer * 5 + inner]
            );
        }
    }
    CHECK(memref_copy_call_count == 2);
    CHECK(memref_copy_byte_counts[0] == 3 * sizeof(float));
    CHECK(memref_copy_byte_counts[1] == 3 * sizeof(float));

    float irregular_source_data[8]{};
    float irregular_destination_data[8]{};
    irregular_source_data[0] = 1.0F;
    irregular_source_data[2] = 2.0F;
    irregular_source_data[5] = 3.0F;
    irregular_source_data[7] = 4.0F;
    RankedMemRef2D irregular_source{
        irregular_source_data,
        irregular_source_data,
        0,
        {2, 2},
        {5, 2},
    };
    RankedMemRef2D irregular_destination{
        irregular_destination_data,
        irregular_destination_data,
        0,
        {2, 2},
        {4, 3},
    };
    UnrankedMemRef irregular_source_unranked{2, &irregular_source};
    UnrankedMemRef irregular_destination_unranked{
        2,
        &irregular_destination,
    };
    resetMemRefCopyCalls();
    memrefCopy(
        sizeof(float),
        &irregular_source_unranked,
        &irregular_destination_unranked
    );
    CHECK(irregular_destination_data[0] == 1.0F);
    CHECK(irregular_destination_data[3] == 2.0F);
    CHECK(irregular_destination_data[4] == 3.0F);
    CHECK(irregular_destination_data[7] == 4.0F);
    CHECK(memref_copy_call_count == 4);
    for (size_t call = 0; call < 4; ++call) {
        CHECK(memref_copy_byte_counts[call] == sizeof(float));
    }
}

void testTaskRegistry() {
    const Task tasks[] = {
        {2, executeNoop, 1, 1},
        {7, executeNoop, 2, 1},
        {42, executeNoop, 1, 2},
    };
    const TaskRegistry registry{tasks, 3};

    CHECK(registry.valid());
    CHECK(registry.find(2) == &tasks[0]);
    CHECK(registry.find(7) == &tasks[1]);
    CHECK(registry.find(42) == &tasks[2]);
    CHECK(registry.find(0) == nullptr);
    CHECK(registry.find(8) == nullptr);

    const Task duplicate[] = {
        {2, executeNoop, 1, 1},
        {2, executeNoop, 1, 1},
    };
    CHECK(!(TaskRegistry{duplicate, 2}.valid()));

    const Task unsorted[] = {
        {9, executeNoop, 1, 1},
        {3, executeNoop, 1, 1},
    };
    CHECK(!(TaskRegistry{unsorted, 2}.valid()));

    const Task no_execute[] = {{1, nullptr, 0, 0}};
    CHECK(!(TaskRegistry{no_execute, 1}.valid()));
    CHECK(!(TaskRegistry{nullptr, 1}.valid()));
    CHECK((TaskRegistry{nullptr, 1}.find(1) == nullptr));
    CHECK((TaskRegistry{nullptr, 0}.valid()));
}

void testTaskInstancePool() {
    TaskInstancePool pool;
    pool.initialize();

    std::array<Tensor, kTaskInstanceCapacity> inputs{};
    std::array<Tensor, kTaskInstanceCapacity> outputs{};
    std::array<TaskInstance*, kTaskInstanceCapacity> instances{};

    for (uint32_t index = 0; index < kTaskInstanceCapacity; ++index) {
        CHECK(pool.get(index) != nullptr);
        CHECK(pool.get(index)->state == TaskInstanceState::Free);
        instances[index] = pool.acquire(
            static_cast<ExecutionId>(index / 2U),
            100U + index,
            &inputs[index],
            &outputs[index]
        );
        CHECK(instances[index] != nullptr);
        CHECK(pool.indexOf(instances[index]) == index);
        CHECK(instances[index]->state == TaskInstanceState::WaitingForInputs);
    }

    CHECK(pool.get(kTaskInstanceCapacity) == nullptr);
    CHECK(pool.acquire(99, 99, nullptr, nullptr) == nullptr);
    CHECK(pool.acquire(0, 100, nullptr, nullptr) == nullptr);
    CHECK(pool.find(3, 106) == instances[6]);
    CHECK(pool.find(100, 106) == nullptr);
    CHECK(!pool.release(instances[0]));

    instances[0]->state = TaskInstanceState::Complete;
    CHECK(pool.release(instances[0]));
    CHECK(instances[0]->state == TaskInstanceState::Free);
    CHECK(pool.find(0, 100) == nullptr);

    TaskInstance* replacement = pool.acquire(77, 900, nullptr, nullptr);
    CHECK(replacement == instances[0]);
    replacement->state = TaskInstanceState::Failed;
    CHECK(pool.release(replacement));
    CHECK(!pool.release(nullptr));
}

void testReadyQueue() {
    ReadyQueue queue;
    queue.initialize();

    CHECK(queue.empty());
    CHECK(!queue.full());
    CHECK(queue.size() == 0);

    uint32_t value = 0;
    CHECK(!queue.pop(&value));
    CHECK(!queue.pop(nullptr));
    CHECK(!queue.push(kTaskInstanceCapacity));

    for (uint32_t index = 0; index < kTaskInstanceCapacity; ++index) {
        CHECK(queue.push(index));
    }
    CHECK(queue.full());
    CHECK(queue.size() == kTaskInstanceCapacity);
    CHECK(!queue.push(0));

    for (uint32_t index = 0; index < 8; ++index) {
        CHECK(queue.pop(&value));
        CHECK(value == index);
    }
    for (uint32_t index = 0; index < 8; ++index) {
        CHECK(queue.push(index));
    }
    for (uint32_t index = 8; index < kTaskInstanceCapacity; ++index) {
        CHECK(queue.pop(&value));
        CHECK(value == index);
    }
    for (uint32_t index = 0; index < 8; ++index) {
        CHECK(queue.pop(&value));
        CHECK(value == index);
    }
    CHECK(queue.empty());
}

struct FakeTransport {
    uint32_t destination;
    uint32_t sent_word;
    uint32_t received_word;
    bool accept_send;
    bool have_receive;
};

bool fakeSend(void* context, uint32_t destination, uint32_t word) {
    auto* fake = static_cast<FakeTransport*>(context);
    if (!fake->accept_send) {
        return false;
    }
    fake->destination = destination;
    fake->sent_word = word;
    return true;
}

bool fakeReceive(void* context, uint32_t* word) {
    auto* fake = static_cast<FakeTransport*>(context);
    if (!fake->have_receive) {
        return false;
    }
    *word = fake->received_word;
    fake->have_receive = false;
    return true;
}

void testTransport() {
    FakeTransport fake{0, 0, UINT32_C(0xdeadbeef), false, true};
    WordTransport transport{&fake, fakeSend, fakeReceive};

    CHECK(transport.valid());
    CHECK(!transport.trySend(8, 9));
    fake.accept_send = true;
    CHECK(transport.trySend(8, 9));
    CHECK(fake.destination == 8);
    CHECK(fake.sent_word == 9);

    uint32_t word = 0;
    CHECK(transport.tryReceive(&word));
    CHECK(word == UINT32_C(0xdeadbeef));
    CHECK(!transport.tryReceive(&word));
    CHECK(!transport.tryReceive(nullptr));

    const WordTransport invalid{nullptr, nullptr, nullptr};
    CHECK(!invalid.valid());
    CHECK(!invalid.trySend(0, 0));
    CHECK(!invalid.tryReceive(&word));
}

struct StreamTransport {
    std::array<uint32_t, 4> sent_words{};
    std::array<uint32_t, 4> received_words{};
    uint32_t sent_count = 0;
    uint32_t received_count = 0;
    uint32_t destination = 0;
};

bool streamSend(void* context, uint32_t destination, uint32_t word) {
    auto* stream = static_cast<StreamTransport*>(context);
    if (stream->sent_count >= stream->sent_words.size()) {
        return false;
    }
    stream->destination = destination;
    stream->sent_words[stream->sent_count++] = word;
    return true;
}

bool streamReceive(void* context, uint32_t* word) {
    auto* stream = static_cast<StreamTransport*>(context);
    if (stream->received_count >= stream->received_words.size()) {
        return false;
    }
    *word = stream->received_words[stream->received_count++];
    return true;
}

void testBasicTileRuntime() {
    const Task boot_tasks[] = {
        {10, executeBoot, 0, 0},
    };
    const Task dispatch_tasks[] = {
        {11, executeAddOne, 1, 1},
    };
    const Route incoming_routes[] = {
        {6, 2, 9, 0, 4, 11, 0, 20, 0, 8},
    };
    const Route outgoing_routes[] = {
        {7, 4, 11, 0, 8, 12, 0, 21, 1, 8},
    };
    const ModelIO model_inputs[] = {
        {0, 4, 18, 2, 4},
    };
    const ModelIO model_outputs[] = {
        {0, 4, 19, 3, 4},
    };
    const TileABI abi{
        4,
        boot_tasks,
        1,
        {dispatch_tasks, 1},
        incoming_routes,
        1,
        outgoing_routes,
        1,
        model_inputs,
        1,
        model_outputs,
        1,
        nullptr,
        0,
        nullptr,
        0,
        0,
        nullptr,
        0,
        nullptr,
        0,
    };

    CHECK(abi.valid());
    CHECK(abi.findIncomingRoute(6) == &incoming_routes[0]);
    CHECK(abi.findIncomingRoute(7) == nullptr);
    CHECK(abi.findOutgoingRoute(7) == &outgoing_routes[0]);
    CHECK(abi.findOutgoingRoute(6) == nullptr);
    CHECK(!abi.hasDeploymentPlan());

    const int64_t dimensions[] = {1, 3, 4, 4, 2};
    const Resource resources[] = {
        {
            18,
            0,
            0,
            ResourceKind::ModelInput,
            ElementType::Float32,
            4,
            0,
            ResourceExternal,
            192,
            0,
        },
        {
            20,
            6,
            1,
            ResourceKind::RouteInput,
            ElementType::Float32,
            1,
            4,
            ResourceWorkspace,
            8,
            0,
        },
        {
            21,
            7,
            2,
            ResourceKind::RouteOutput,
            ElementType::Float32,
            1,
            4,
            ResourceWorkspace,
            8,
            8,
        },
    };
    const uint32_t binding_data[] = {1, 2};
    const TaskBinding task_bindings[] = {
        {11, 0, 1, 1, 1, 2, 0, 0},
    };
    TileABI deployment_abi = abi;
    deployment_abi.resources = resources;
    deployment_abi.resource_count = 3;
    deployment_abi.resource_dimensions = dimensions;
    deployment_abi.resource_dimension_count = 5;
    deployment_abi.workspace_size = 16;
    deployment_abi.task_bindings = task_bindings;
    deployment_abi.task_binding_count = 1;
    deployment_abi.task_binding_data = binding_data;
    deployment_abi.task_binding_data_count = 2;
    CHECK(deployment_abi.hasDeploymentPlan());
    CHECK(deployment_abi.validDeploymentPlan());
    CHECK(deployment_abi.valid());
    CHECK(deployment_abi.findResource(1) == &resources[1]);
    CHECK(deployment_abi.findResource(3) == nullptr);
    CHECK(deployment_abi.findTaskBinding(11) == &task_bindings[0]);
    CHECK(deployment_abi.findTaskBinding(99) == nullptr);

    StreamTransport stream{
        {},
        {UINT32_C(0x11223344), UINT32_C(0xaabbccdd), 0, 0},
    };
    const WordTransport transport{&stream, streamSend, streamReceive};
    BasicTileRuntime runtime{abi, transport};

    float input_value = 5.0F;
    float output_value = 0.0F;
    Tensor inputs[] = {
        {ElementType::Float32, 0, &input_value},
    };
    Tensor outputs[] = {
        {ElementType::Float32, 0, &output_value},
    };

    CHECK(runtime.valid());
    CHECK(!runtime.booted());
    CHECK(!runtime.execute(11, inputs, 1, outputs, 1));
    boot_count = 0;
    CHECK(runtime.boot());
    CHECK(runtime.booted());
    CHECK(boot_count == 1);
    CHECK(runtime.boot());
    CHECK(boot_count == 1);
    CHECK(!runtime.execute(99, inputs, 1, outputs, 1));
    CHECK(!runtime.execute(11, inputs, 0, outputs, 1));
    CHECK(runtime.execute(11, inputs, 1, outputs, 1));
    CHECK(output_value == 6.0F);

    const uint32_t outgoing[] = {
        UINT32_C(0x01020304),
        UINT32_C(0xdeadbeef),
    };
    CHECK(!runtime.sendRoute(99, outgoing, sizeof(outgoing)));
    CHECK(!runtime.sendRoute(7, outgoing, sizeof(uint32_t)));
    CHECK(runtime.sendRoute(7, outgoing, sizeof(outgoing)));
    CHECK(stream.destination == 8);
    CHECK(stream.sent_count == 2);
    CHECK(stream.sent_words[0] == outgoing[0]);
    CHECK(stream.sent_words[1] == outgoing[1]);

    uint32_t incoming[2]{};
    CHECK(!runtime.receiveRoute(99, incoming, sizeof(incoming)));
    CHECK(runtime.receiveRoute(6, incoming, sizeof(incoming)));
    CHECK(incoming[0] == UINT32_C(0x11223344));
    CHECK(incoming[1] == UINT32_C(0xaabbccdd));

    Route invalid_route = outgoing_routes[0];
    invalid_route.byte_size = 6;
    TileABI invalid_abi = abi;
    invalid_abi.outgoing_routes = &invalid_route;
    CHECK(!invalid_abi.valid());

    Resource invalid_resource = resources[1];
    invalid_resource.workspace_offset = 12;
    deployment_abi.resources = &invalid_resource;
    deployment_abi.resource_count = 1;
    CHECK(!deployment_abi.validDeploymentPlan());
}

struct RoutedStreamTransport {
    std::array<uint32_t, 8> sent_words{};
    std::array<RoutedWord, 8> received_words{};
    uint32_t sent_count = 0;
    uint32_t received_count = 0;
    uint32_t available_receive_count = 0;
    uint32_t destination = 0;
    bool send_blocked = false;
};

bool routedStreamSend(
    void* context,
    uint32_t destination,
    uint32_t word
) {
    auto* stream = static_cast<RoutedStreamTransport*>(context);
    if (stream->send_blocked ||
        stream->sent_count >= stream->sent_words.size()) {
        return false;
    }
    stream->destination = destination;
    stream->sent_words[stream->sent_count++] = word;
    return true;
}

bool routedStreamSendWords(
    void* context,
    uint32_t destination,
    const uint32_t* words,
    uint32_t word_count
) {
    auto* stream = static_cast<RoutedStreamTransport*>(context);
    if (stream->send_blocked ||
        words == nullptr ||
        word_count == 0 ||
        word_count > stream->sent_words.size() - stream->sent_count) {
        return false;
    }
    stream->destination = destination;
    for (uint32_t index = 0; index < word_count; ++index) {
        stream->sent_words[stream->sent_count++] = words[index];
    }
    return true;
}

bool routedStreamReceive(void* context, RoutedWord* word) {
    auto* stream = static_cast<RoutedStreamTransport*>(context);
    if (stream->received_count >= stream->available_receive_count) {
        return false;
    }
    *word = stream->received_words[stream->received_count++];
    return true;
}

struct RecordedTaskTrace {
    TaskTraceEvent event;
    uint32_t task_id;
    ExecutionId execution_id;
};

struct TaskTraceRecorder {
    std::array<RecordedTaskTrace, 4> events{};
    uint32_t count = 0;
};

void recordTaskTrace(
    void* context,
    TaskTraceEvent event,
    uint32_t task_id,
    ExecutionId execution_id
) {
    auto* recorder = static_cast<TaskTraceRecorder*>(context);
    if (recorder == nullptr ||
        recorder->count >= recorder->events.size()) {
        return;
    }
    recorder->events[recorder->count++] = {
        event,
        task_id,
        execution_id,
    };
}

void testDeploymentRuntime() {
    const Task source_tasks[] = {
        {11, executeMemRefAddOne, 1, 1},
    };
    const Route outgoing_routes[] = {
        {7, 4, 11, 0, 8, 12, 0, 2, 1, 4},
    };
    const ModelIO model_inputs[] = {
        {0, 4, 1, 0, 4},
    };
    const Resource source_resources[] = {
        {
            1,
            0,
            0,
            ResourceKind::ModelInput,
            ElementType::Float32,
            0,
            0,
            ResourceExternal,
            4,
            0,
        },
        {
            2,
            7,
            1,
            ResourceKind::RouteOutput,
            ElementType::Float32,
            0,
            0,
            ResourceWorkspace,
            4,
            0,
        },
    };
    const uint32_t source_binding_data[] = {0, 1};
    const TaskBinding source_bindings[] = {
        {11, 0, 1, 1, 1, 2, 0, 0},
    };
    const TileABI source_abi{
        4,
        nullptr,
        0,
        {source_tasks, 1},
        nullptr,
        0,
        outgoing_routes,
        1,
        model_inputs,
        1,
        nullptr,
        0,
        source_resources,
        2,
        nullptr,
        0,
        4,
        source_bindings,
        1,
        source_binding_data,
        2,
    };

    RoutedStreamTransport source_stream{};
    const RoutedWordTransport source_transport{
        &source_stream,
        routedStreamSend,
        routedStreamReceive,
        routedStreamSendWords,
    };
    TaskTraceRecorder trace_recorder{};
    DeploymentTrace source_trace{
        &trace_recorder,
        recordTaskTrace,
    };
    DeploymentRuntime source_runtime{
        source_abi,
        source_transport,
        nullptr,
        &source_trace,
    };
    float source_input = 5.0F;
    CHECK(source_runtime.initialize());
    CHECK(source_runtime.bindModelInput(0, &source_input));
    CHECK(source_runtime.step() == DeploymentStep::Progress);
    source_stream.send_blocked = true;
    CHECK(source_runtime.step() == DeploymentStep::WaitForTransmit);
    source_stream.send_blocked = false;
    CHECK(source_runtime.step() == DeploymentStep::Progress);
    CHECK(source_runtime.step() == DeploymentStep::Progress);
    CHECK(source_runtime.step() == DeploymentStep::Complete);
    CHECK(source_runtime.complete());
    CHECK(!source_runtime.failed());
    CHECK(source_stream.destination == 8);
    CHECK(source_stream.sent_count == 6);
    CHECK(source_stream.sent_words[0] == UINT32_C(0x474f4c4d));
    CHECK(source_stream.sent_words[1] == 7);
    CHECK(source_stream.sent_words[2] == 0);
    CHECK(source_stream.sent_words[3] == 0);
    CHECK(source_stream.sent_words[4] == 1);
    CHECK(source_stream.sent_words[5] == std::bit_cast<uint32_t>(6.0F));
    CHECK(trace_recorder.count == 2);
    CHECK(
        trace_recorder.events[0].event ==
        TaskTraceEvent::Start
    );
    CHECK(trace_recorder.events[0].task_id == 11);
    CHECK(trace_recorder.events[0].execution_id == 0);
    CHECK(
        trace_recorder.events[1].event ==
        TaskTraceEvent::Finish
    );
    CHECK(trace_recorder.events[1].task_id == 11);
    CHECK(trace_recorder.events[1].execution_id == 0);

    const Task destination_tasks[] = {
        {12, executeMemRefAddOne, 1, 1},
    };
    const Route incoming_routes[] = {
        {7, 4, 11, 0, 8, 12, 0, 2, 0, 4},
    };
    const ModelIO model_outputs[] = {
        {0, 8, 3, 1, 4},
    };
    const Resource destination_resources[] = {
        {
            2,
            7,
            0,
            ResourceKind::RouteInput,
            ElementType::Float32,
            0,
            0,
            ResourceWorkspace,
            4,
            0,
        },
        {
            3,
            0,
            1,
            ResourceKind::ModelOutput,
            ElementType::Float32,
            0,
            0,
            ResourceExternal,
            4,
            0,
        },
    };
    const uint32_t destination_binding_data[] = {0, 1};
    const TaskBinding destination_bindings[] = {
        {12, 0, 1, 1, 1, 2, 0, 0},
    };
    const TileABI destination_abi{
        8,
        nullptr,
        0,
        {destination_tasks, 1},
        incoming_routes,
        1,
        nullptr,
        0,
        nullptr,
        0,
        model_outputs,
        1,
        destination_resources,
        2,
        nullptr,
        0,
        4,
        destination_bindings,
        1,
        destination_binding_data,
        2,
    };

    RoutedStreamTransport destination_stream{};
    destination_stream.received_words = {
        RoutedWord{4, UINT32_C(0x474f4c4d)},
        RoutedWord{4, 7},
        RoutedWord{4, 0},
        RoutedWord{4, 0},
        RoutedWord{4, 1},
        RoutedWord{4, std::bit_cast<uint32_t>(6.0F)},
        RoutedWord{},
        RoutedWord{},
    };
    const RoutedWordTransport destination_transport{
        &destination_stream,
        routedStreamSend,
        routedStreamReceive,
    };
    DeploymentRuntime destination_runtime{
        destination_abi,
        destination_transport,
    };
    float destination_output = 0.0F;
    CHECK(destination_runtime.initialize());
    CHECK(destination_runtime.bindModelOutput(0, &destination_output));
    for (uint32_t index = 0; index < 6; ++index) {
        CHECK(
            destination_runtime.step() ==
            DeploymentStep::WaitForReceive
        );
        destination_stream.available_receive_count = index + 1;
        CHECK(destination_runtime.step() == DeploymentStep::Progress);
    }
    CHECK(destination_runtime.step() == DeploymentStep::Complete);
    CHECK(destination_runtime.complete());
    CHECK(destination_output == 7.0F);

    RoutedStreamTransport invalid_stream{};
    invalid_stream.received_words = {
        RoutedWord{4, UINT32_C(0xdeadbeef)},
    };
    invalid_stream.available_receive_count = 1;
    const RoutedWordTransport invalid_transport{
        &invalid_stream,
        routedStreamSend,
        routedStreamReceive,
    };
    DeploymentRuntime invalid_runtime{
        destination_abi,
        invalid_transport,
    };
    float invalid_output = 0.0F;
    CHECK(invalid_runtime.initialize());
    CHECK(invalid_runtime.bindModelOutput(0, &invalid_output));
    CHECK(invalid_runtime.step() == DeploymentStep::Failed);
    CHECK(invalid_runtime.error() == DeploymentError::InvalidFrame);
    const InvalidFrameDiagnostic& diagnostic =
        invalid_runtime.invalidFrameDiagnostic();
    CHECK(diagnostic.reason == InvalidFrameReason::InvalidMagic);
    CHECK(diagnostic.source_tile == 4);
    CHECK(diagnostic.expected == UINT32_C(0x474f4c4d));
    CHECK(diagnostic.actual == UINT32_C(0xdeadbeef));
}

struct ReceiveDMATransport {
    std::array<RoutedWord, 16> headers{};
    std::array<uint32_t, 4> completion_sources{};
    std::array<uint32_t, 4> completion_routes{};
    uint32_t header_count = 0;
    uint32_t header_index = 0;
    uint32_t completion_read = 0;
    uint32_t completion_write = 0;
    uint32_t start_count = 0;
};

bool receiveDMASend(void*, uint32_t, uint32_t) {
    return false;
}

bool receiveDMAHeader(void* context, RoutedWord* word) {
    auto* transport =
        static_cast<ReceiveDMATransport*>(context);
    if (transport == nullptr ||
        word == nullptr ||
        transport->header_index >= transport->header_count) {
        return false;
    }
    *word = transport->headers[transport->header_index++];
    return true;
}

bool receiveDMAStart(
    void* context,
    uint32_t source,
    uint32_t route_id,
    void* destination,
    uint32_t word_count
) {
    auto* transport =
        static_cast<ReceiveDMATransport*>(context);
    if (transport == nullptr ||
        destination == nullptr ||
        word_count != 1 ||
        transport->completion_write >=
            transport->completion_sources.size()) {
        return false;
    }
    float value = 0.0F;
    if (source == 4 && route_id == 7) {
        value = 3.0F;
    } else if (source == 5 && route_id == 8) {
        value = 4.0F;
    } else {
        return false;
    }
    *static_cast<float*>(destination) = value;
    const uint32_t completion = transport->completion_write++;
    transport->completion_sources[completion] = source;
    transport->completion_routes[completion] = route_id;
    ++transport->start_count;
    return true;
}

bool receiveDMACompletion(
    void* context,
    uint32_t* source,
    uint32_t* route_id
) {
    auto* transport =
        static_cast<ReceiveDMATransport*>(context);
    if (transport == nullptr ||
        source == nullptr ||
        route_id == nullptr ||
        transport->completion_read >=
            transport->completion_write) {
        return false;
    }
    const uint32_t completion = transport->completion_read++;
    *source = transport->completion_sources[completion];
    *route_id = transport->completion_routes[completion];
    return true;
}

void testDeploymentReceiveDMA() {
    const Task tasks[] = {
        {12, executeMemRefAddTwo, 2, 1},
    };
    const Route incoming_routes[] = {
        {7, 4, 40, 0, 8, 12, 0, 2, 0, 4},
        {8, 5, 50, 0, 8, 12, 1, 3, 1, 4},
    };
    const ModelIO model_outputs[] = {
        {0, 8, 4, 2, 4},
    };
    const Resource resources[] = {
        {
            2,
            7,
            0,
            ResourceKind::RouteInput,
            ElementType::Float32,
            0,
            0,
            ResourceWorkspace,
            4,
            0,
        },
        {
            3,
            8,
            1,
            ResourceKind::RouteInput,
            ElementType::Float32,
            0,
            0,
            ResourceWorkspace,
            4,
            4,
        },
        {
            4,
            0,
            2,
            ResourceKind::ModelOutput,
            ElementType::Float32,
            0,
            0,
            ResourceExternal,
            4,
            0,
        },
    };
    const uint32_t binding_data[] = {0, 1, 2};
    const TaskBinding bindings[] = {
        {12, 0, 2, 2, 1, 3, 0, 0},
    };
    const TileABI abi{
        8,
        nullptr,
        0,
        {tasks, 1},
        incoming_routes,
        2,
        nullptr,
        0,
        nullptr,
        0,
        model_outputs,
        1,
        resources,
        3,
        nullptr,
        0,
        8,
        bindings,
        1,
        binding_data,
        3,
    };

    ReceiveDMATransport stream{};
    stream.headers = {
        RoutedWord{4, UINT32_C(0x474f4c4d)},
        RoutedWord{5, UINT32_C(0x474f4c4d)},
        RoutedWord{4, 7},
        RoutedWord{5, 8},
        RoutedWord{4, 0},
        RoutedWord{5, 0},
        RoutedWord{4, 0},
        RoutedWord{5, 0},
        RoutedWord{4, 1},
        RoutedWord{5, 1},
    };
    stream.header_count = 10;
    const RoutedWordTransport transport{
        &stream,
        receiveDMASend,
        receiveDMAHeader,
        nullptr,
        receiveDMAStart,
        receiveDMACompletion,
    };
    DeploymentRuntime runtime{abi, transport};
    float output = 0.0F;
    CHECK(runtime.initialize());
    CHECK(runtime.bindModelOutput(0, &output));
    for (uint32_t step = 0;
         step < 100 && !runtime.complete() && !runtime.failed();
         ++step) {
        runtime.step();
    }
    CHECK(!runtime.failed());
    CHECK(runtime.complete());
    CHECK(output == 7.0F);
    CHECK(stream.header_index == 10);
    CHECK(stream.start_count == 2);
    CHECK(stream.completion_read == 2);
}

struct ReciprocalCell {
    std::array<uint32_t, 8> words{};
    uint32_t source = 0;
    uint32_t word_count = 0;
    uint32_t read_index = 0;
    bool occupied = false;
};

struct ReciprocalLink {
    ReciprocalCell incoming[2];
};

struct ReciprocalEndpoint {
    ReciprocalLink* link;
    uint32_t tile;
};

bool reciprocalSendWords(
    void* context,
    uint32_t destination,
    const uint32_t* words,
    uint32_t word_count
);

bool reciprocalSend(
    void* context,
    uint32_t destination,
    uint32_t word
) {
    return reciprocalSendWords(
        context, destination, &word, 1);
}

bool reciprocalSendWords(
    void* context,
    uint32_t destination,
    const uint32_t* words,
    uint32_t word_count
) {
    auto* endpoint = static_cast<ReciprocalEndpoint*>(context);
    if (endpoint == nullptr ||
        endpoint->link == nullptr ||
        destination >= 2 ||
        destination == endpoint->tile ||
        words == nullptr ||
        word_count == 0 ||
        word_count > endpoint->link->incoming[destination].words.size()) {
        return false;
    }
    ReciprocalCell& cell = endpoint->link->incoming[destination];
    if (cell.occupied) {
        return false;
    }
    for (uint32_t index = 0; index < word_count; ++index) {
        cell.words[index] = words[index];
    }
    cell.source = endpoint->tile;
    cell.word_count = word_count;
    cell.read_index = 0;
    cell.occupied = true;
    return true;
}

bool reciprocalReceive(void* context, RoutedWord* word) {
    auto* endpoint = static_cast<ReciprocalEndpoint*>(context);
    if (endpoint == nullptr ||
        endpoint->link == nullptr ||
        endpoint->tile >= 2 ||
        word == nullptr) {
        return false;
    }
    ReciprocalCell& cell = endpoint->link->incoming[endpoint->tile];
    if (!cell.occupied) {
        return false;
    }
    *word = {cell.source, cell.words[cell.read_index++]};
    if (cell.read_index == cell.word_count) {
        cell.occupied = false;
    }
    return true;
}

void testReciprocalDeploymentProgress() {
    const Task tile0_tasks[] = {
        {10, executeMemRefAddOne, 1, 1},
        {20, executeMemRefAddOne, 1, 1},
    };
    const Route tile0_incoming[] = {
        {101, 1, 11, 0, 0, 20, 0, 102, 2, 4},
    };
    const Route tile0_outgoing[] = {
        {100, 0, 10, 0, 1, 21, 0, 101, 1, 4},
    };
    const ModelIO tile0_inputs[] = {
        {0, 0, 1, 0, 4},
    };
    const ModelIO tile0_outputs[] = {
        {0, 0, 3, 3, 4},
    };
    const Resource tile0_resources[] = {
        {1, 0, 0, ResourceKind::ModelInput, ElementType::Float32,
         0, 0, ResourceExternal, 4, 0},
        {101, 100, 1, ResourceKind::RouteOutput, ElementType::Float32,
         0, 0, ResourceWorkspace, 4, 0},
        {102, 101, 2, ResourceKind::RouteInput, ElementType::Float32,
         0, 0, ResourceWorkspace, 4, 4},
        {3, 0, 3, ResourceKind::ModelOutput, ElementType::Float32,
         0, 0, ResourceExternal, 4, 0},
    };
    const uint32_t tile0_binding_data[] = {0, 1, 2, 3};
    const TaskBinding tile0_bindings[] = {
        {10, 0, 1, 1, 1, 0, 0, 0},
        {20, 2, 1, 3, 1, 0, 0, 0},
    };
    const TileABI tile0_abi{
        0, nullptr, 0, {tile0_tasks, 2},
        tile0_incoming, 1, tile0_outgoing, 1,
        tile0_inputs, 1, tile0_outputs, 1,
        tile0_resources, 4, nullptr, 0, 8,
        tile0_bindings, 2, tile0_binding_data, 4,
    };

    const Task tile1_tasks[] = {
        {11, executeMemRefAddOne, 1, 1},
        {21, executeMemRefAddOne, 1, 1},
    };
    const Route tile1_incoming[] = {
        {100, 0, 10, 0, 1, 21, 0, 101, 2, 4},
    };
    const Route tile1_outgoing[] = {
        {101, 1, 11, 0, 0, 20, 0, 102, 1, 4},
    };
    const ModelIO tile1_inputs[] = {
        {0, 1, 2, 0, 4},
    };
    const ModelIO tile1_outputs[] = {
        {0, 1, 4, 3, 4},
    };
    const Resource tile1_resources[] = {
        {2, 0, 0, ResourceKind::ModelInput, ElementType::Float32,
         0, 0, ResourceExternal, 4, 0},
        {102, 101, 1, ResourceKind::RouteOutput, ElementType::Float32,
         0, 0, ResourceWorkspace, 4, 0},
        {101, 100, 2, ResourceKind::RouteInput, ElementType::Float32,
         0, 0, ResourceWorkspace, 4, 4},
        {4, 0, 3, ResourceKind::ModelOutput, ElementType::Float32,
         0, 0, ResourceExternal, 4, 0},
    };
    const uint32_t tile1_binding_data[] = {0, 1, 2, 3};
    const TaskBinding tile1_bindings[] = {
        {11, 0, 1, 1, 1, 0, 0, 0},
        {21, 2, 1, 3, 1, 0, 0, 0},
    };
    const TileABI tile1_abi{
        1, nullptr, 0, {tile1_tasks, 2},
        tile1_incoming, 1, tile1_outgoing, 1,
        tile1_inputs, 1, tile1_outputs, 1,
        tile1_resources, 4, nullptr, 0, 8,
        tile1_bindings, 2, tile1_binding_data, 4,
    };

    ReciprocalLink link{};
    ReciprocalEndpoint tile0_endpoint{&link, 0};
    ReciprocalEndpoint tile1_endpoint{&link, 1};
    const RoutedWordTransport tile0_transport{
        &tile0_endpoint,
        reciprocalSend,
        reciprocalReceive,
        reciprocalSendWords,
    };
    const RoutedWordTransport tile1_transport{
        &tile1_endpoint,
        reciprocalSend,
        reciprocalReceive,
        reciprocalSendWords,
    };
    DeploymentRuntime tile0_runtime{tile0_abi, tile0_transport};
    DeploymentRuntime tile1_runtime{tile1_abi, tile1_transport};
    float tile0_input = 1.0F;
    float tile1_input = 10.0F;
    float tile0_output = 0.0F;
    float tile1_output = 0.0F;

    CHECK(tile0_runtime.bindModelInput(0, &tile0_input));
    CHECK(tile0_runtime.bindModelOutput(0, &tile0_output));
    CHECK(tile1_runtime.bindModelInput(0, &tile1_input));
    CHECK(tile1_runtime.bindModelOutput(0, &tile1_output));

    for (uint32_t step = 0;
         step < 100 &&
         (!tile0_runtime.complete() || !tile1_runtime.complete());
         ++step) {
        tile0_runtime.step();
        tile1_runtime.step();
    }

    CHECK(!tile0_runtime.failed());
    CHECK(!tile1_runtime.failed());
    CHECK(tile0_runtime.complete());
    CHECK(tile1_runtime.complete());
    CHECK(tile0_output == 12.0F);
    CHECK(tile1_output == 3.0F);
}

}  // namespace

int main() {
    testTensor();
    testScratchpadABI();
    testMemRefCopy();
    testTaskRegistry();
    testTaskInstancePool();
    testReadyQueue();
    testTransport();
    testBasicTileRuntime();
    testDeploymentRuntime();
    testDeploymentReceiveDMA();
    testReciprocalDeploymentProgress();

    if (failures != 0) {
        std::fprintf(stderr, "%d runtime checks failed\n", failures);
        return 1;
    }

    std::puts("runtime unit tests: PASS");
    return 0;
}
