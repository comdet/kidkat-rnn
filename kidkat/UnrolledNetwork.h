#ifndef TINYRNN_VMNETWORK_H_INCLUDED
#define TINYRNN_VMNETWORK_H_INCLUDED

#include "Common.h"
#include "UnrolledNeuron.h"
#include "UnrolledTrainingContext.h"
#include "ScopedMemoryBlock.h"
#include "ScopedTimer.h"
#include "SerializationKeys.h"
#include <ctime>
#include <cstdlib>


class UnrolledNetwork final
{
public:
    
    using Ptr = std::shared_ptr<UnrolledNetwork>;
    using VMLayers = std::vector<UnrolledNeuron::Vector>;
    
public:
    
    explicit UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext);
    UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext, VMLayers targetLayers);
    
    UnrolledTrainingContext::Ptr getContext() const noexcept;
    
    UnrolledTrainingContext::RawData feed(const UnrolledTrainingContext::RawData &values);
    void train(Value rate, const UnrolledTrainingContext::RawData &target);        
    
private:
    
    UnrolledTrainingContext::Ptr trainingContext;
    
private:
  
    class Kernel final
    {
    public:
        
        using Ptr = std::shared_ptr<Kernel>;
        Kernel() = default;
        
        std::vector<char> commands;
        std::vector<Index> indices; // Index is the same type as cl_uint
        
    private:
        
        TINYRNN_DISALLOW_COPY_AND_ASSIGN(Kernel);
    };
    
    Kernel::Ptr feedKernel;
    Kernel::Ptr trainKernel;
    
    Kernel::Ptr compileFeedKernel(const VMLayers &targetLayers) const;
    Kernel::Ptr compileTrainKernel(const VMLayers &targetLayers) const;
    
    bool initialize(const VMLayers &targetLayers);
    
    TINYRNN_DISALLOW_COPY_AND_ASSIGN(UnrolledNetwork);
};

//===------------------------------------------------------------------===//
// HardcodedNetwork implementation
//===------------------------------------------------------------------===//

inline UnrolledNetwork::UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext) :
trainingContext(targetContext)
{
    VMLayers empty;
    this->initialize(empty);
}

inline UnrolledNetwork::UnrolledNetwork(UnrolledTrainingContext::Ptr targetContext,
                            VMLayers targetLayers) :
trainingContext(targetContext)
{
    this->initialize(targetLayers);
}

inline UnrolledTrainingContext::Ptr UnrolledNetwork::getContext() const noexcept
{
    return this->trainingContext;
}

//===------------------------------------------------------------------===//
// Compiling
//===------------------------------------------------------------------===//

inline bool UnrolledNetwork::initialize(const VMLayers &targetLayers)
{
    const ScopedTimer timer("UnrolledNetwork::initialize");
    
    srand(time(0));
    
    this->feedKernel = this->compileFeedKernel(targetLayers);
    this->trainKernel = this->compileTrainKernel(targetLayers);
    
    return true;
}

//#define VALUE_STRING std::string((sizeof(Value) == sizeof(double)) ? "double" : "float")

//===------------------------------------------------------------------===//
// Compiling all the expressions
//===------------------------------------------------------------------===//

static bool kVMUsesDropout = true;

static void vmProcess(const char *commands,
                      const Index *indices,
                      Value *registers)
{
    uint32_t c = 0; // command number
    uint32_t i = 0; // index number
    char command = 0;
    
#define I(INDEX) (indices[i + INDEX])
#define X(INDEX) (registers[indices[i + INDEX]])
#define SKIP(NUMBER) (i += NUMBER)
    
    // At test time, do not droupout,
    // But scale activation by p (the probability of dropout, 0.5 for now)
    const Value dropout = kVMUsesDropout ? Value(rand() % 2) : Value(0.5);
    
    while (command != VMProgram::End)
    {
        switch (command = commands[c++])
        {
            case VMProgram::Zero:
                X(0) = 0;
                SKIP(1);
                break;
            case VMProgram::Clip:
                X(0) = std::max(Value(-TINYRNN_GRADIENT_CLIPPING_THRESHOLD),
                                std::min(X(0), Value(TINYRNN_GRADIENT_CLIPPING_THRESHOLD)));
                SKIP(1);
                break;
                
            case VMProgram::ActivationSigmoid:
                X(0) = (1.0 / (1.0 + exp(-X(1))));
                i += 2;
                break;
            case VMProgram::DropoutActivationSigmoid:
                X(0) = Value(dropout) * (1.0 / (1.0 + exp(-X(1))));
                i += 2;
                break;
            case VMProgram::DerivativeSigmoid:
                X(0) = X(1) * (1.0 - X(1));
                i += 2;
                break;
                
            case VMProgram::ActivationTanh:
            {
                const Value eP = exp(X(1));
                const Value eN = 1.0 / eP;
                X(0) = (eP - eN) / (eP + eN);
                i += 2;
                break;
            }
            case VMProgram::DropoutActivationTanh:
            {
                const Value eP = exp(X(1));
                const Value eN = 1.0 / eP;
                X(0) = Value(dropout) * ((eP - eN) / (eP + eN));
                i += 2;
                break;
            }
            case VMProgram::DerivativeTanh:
                X(0) = 1.0 - (X(1) * X(1));
                i += 2;
                break;
                
            case VMProgram::ActivationLeakyReLU:
                X(0) = X(1) > 0.0 ? X(1) : (0.01 * X(1));
                SKIP(2);
                break;
            case VMProgram::DropoutActivationLeakyReLU:
                X(0) = Value(dropout) * (X(1) > 0.0 ? X(1) : (0.01 * X(1)));
                SKIP(2);
                break;
            case VMProgram::DerivativeLeakyReLU:
                X(0) = X(1) > 0.0 ? 1.0 : 0.01;
                SKIP(2);
                break;
                
            case VMProgram::AAP:
                X(0) = X(0) + X(1) * X(2);
                SKIP(3);
                break;
            case VMProgram::AAPP:
                X(0) = X(0) + X(1) * X(2) * X(3);
                SKIP(4);
                break;
            case VMProgram::A:
                X(0) = X(1);
                SKIP(2);
                break;
            case VMProgram::AS:
                X(0) = X(1) + X(2);
                SKIP(3);
                break;
            case VMProgram::AD:
                X(0) = X(1) - X(2);
                SKIP(3);
                break;
            case VMProgram::AP:
                X(0) = X(1) * X(2);
                SKIP(3);
                break;
            case VMProgram::APP:
                X(0) = X(1) * X(2) * X(3);
                SKIP(4);
                break;
            case VMProgram::APS:
                X(0) = X(1) * X(2) + X(3);
                SKIP(4);
                break;
            case VMProgram::APSP:
                X(0) = X(1) * X(2) + X(3) * X(4);
                SKIP(5);
                break;
            case VMProgram::APPS:
                X(0) = X(1) * X(2) * X(3) + X(4);
                SKIP(5);
                break;
            case VMProgram::APPSP:
                X(0) = X(1) * X(2) * X(3) + X(4) * X(5);
                SKIP(6);
                break;
            case VMProgram::APPSPP:
                X(0) = X(1) * X(2) * X(3) + X(4) * X(5) * X(6);
                SKIP(7);
                break;
                
            case VMProgram::FeedState:
            {
                const auto loopCount = I(0);
                const auto stateIndex = I(1);
                SKIP(2);
                
                for (Index loop = 0; loop < loopCount; ++loop)
                {
                    registers[stateIndex] = registers[stateIndex] + X(0) * X(1) * X(2);
                    SKIP(3);
                }
                
                break;
            }
                
            default:
                break;
        }
    }
}

inline UnrolledNetwork::Kernel::Ptr
UnrolledNetwork::compileFeedKernel(const VMLayers &targetLayers) const
{
    Kernel::Ptr kernel(new Kernel());
    
    for (const auto &layer : targetLayers)
    {
        for (const auto &neuron : layer)
        {
            const auto &feedCommands = neuron->getFeedChunk().commands;
            const auto &traceCommands = neuron->getTraceChunk().commands;
            kernel->commands.reserve(kernel->commands.size() + feedCommands.size() + traceCommands.size());
            kernel->commands.insert(kernel->commands.end(), feedCommands.begin(), feedCommands.end());
            kernel->commands.insert(kernel->commands.end(), traceCommands.begin(), traceCommands.end());
            
            const auto &feedIndices = neuron->getFeedChunk().indices;
            const auto &traceIndices = neuron->getTraceChunk().indices;
            kernel->indices.reserve(kernel->indices.size() + feedIndices.size() + traceIndices.size());
            kernel->indices.insert(kernel->indices.end(), feedIndices.begin(), feedIndices.end());
            kernel->indices.insert(kernel->indices.end(), traceIndices.begin(), traceIndices.end());
        }
    }
    
    kernel->commands.push_back(VMProgram::End);
    return kernel;
}

inline UnrolledNetwork::Kernel::Ptr
UnrolledNetwork::compileTrainKernel(const VMLayers &targetLayers) const
{
    Kernel::Ptr kernel(new Kernel());
    
    for (size_t l = targetLayers.size(); l --> 0 ;)
    {
        const auto &layer = targetLayers[l];
        
        for (size_t n = layer.size(); n --> 0 ;)
        {
            const auto &neuron = layer[n];
            const auto &trainCommands = neuron->getTrainChunk().commands;
            const auto &trainIndices = neuron->getTrainChunk().indices;
            kernel->commands.reserve(kernel->commands.size() + trainCommands.size());
            kernel->commands.insert(kernel->commands.end(), trainCommands.begin(), trainCommands.end());
            kernel->indices.reserve(kernel->indices.size() + trainIndices.size());
            kernel->indices.insert(kernel->indices.end(), trainIndices.begin(), trainIndices.end());
        }
    }
    
    kernel->commands.push_back(VMProgram::End);
    return kernel;
}

//===------------------------------------------------------------------===//
// Core
//===------------------------------------------------------------------===//

inline UnrolledTrainingContext::RawData UnrolledNetwork::feed(const UnrolledTrainingContext::RawData &inputs)
{
    std::fill(this->trainingContext->getOutputs().begin(),
              this->trainingContext->getOutputs().end(),
              0.0);
    
    const auto &inputIds = this->trainingContext->getInputVariables();
    for (size_t i = 0; i < inputIds.size(); ++i)
    {
        this->trainingContext->getMemory()[inputIds[i]] = inputs[i];
    }
    
    vmProcess(this->feedKernel->commands.data(),
              this->feedKernel->indices.data(),
              this->trainingContext->getMemory().data());
    
    const auto &outputIds = this->trainingContext->getOutputVariables();
    for (size_t i = 0; i < outputIds.size(); ++i)
    {
        this->trainingContext->getOutputs()[i] = this->trainingContext->getMemory()[outputIds[i]];
    }
    
    // Set not to use dropout next time we feed forward
    // Will be reset back to true in train()
    kVMUsesDropout = false;
    
    return this->trainingContext->getOutputs();
}

inline void UnrolledNetwork::train(Value rate, const UnrolledTrainingContext::RawData &targets)
{
    kVMUsesDropout = true;
    
    const auto &targetIds = this->trainingContext->getTargetVariables();
    for (size_t i = 0; i < targetIds.size(); ++i)
    {
        this->trainingContext->getMemory()[targetIds[i]] = targets[i];
    }
    
    const auto rateId = this->trainingContext->getRateVariable();
    this->trainingContext->getMemory()[rateId] = rate;
    
    vmProcess(this->trainKernel->commands.data(),
              this->trainKernel->indices.data(),
              this->trainingContext->getMemory().data());
}

#endif // TINYRNN_VMNETWORK_H_INCLUDED
