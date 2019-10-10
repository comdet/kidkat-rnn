#ifndef TINYRNN_UNROLLEDTRAININGCONTEXT_H_INCLUDED
#define TINYRNN_UNROLLEDTRAININGCONTEXT_H_INCLUDED

#include "Common.h"
#include "Neuron.h"
#include "SerializationKeys.h"

#include <random>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
   
class UnrolledTrainingContext final
{
public:
    
    using Ptr = std::shared_ptr<UnrolledTrainingContext>;
    using RawData = std::vector<Value>;
    using Indices = std::vector<Index>;
    using Mapping = std::map<std::string, Index>;
    using VariableKey = std::vector<Id>;
    
public:
    
    UnrolledTrainingContext();
    
    void restoreNeuronState(Neuron::Ptr targetNeuron);

    Value evaluateVariable(const VariableKey &variableKey, Value defaultValue);
    Index allocateOrReuseVariable(Value value, const VariableKey &variableKey);
    
    void registerInputVariable(Index variableIndex);
    void registerOutputVariable(Index variableIndex);
    void registerTargetVariable(Index variableIndex);
    void registerRateVariable(Index variableIndex);
    
    Indices getInputVariables() const;
    Indices getOutputVariables() const;
    Indices getTargetVariables() const;
    Index getRateVariable() const;
    
    RawData &getMemory();
    RawData &getOutputs();
    
    void clear();
    void clearMappings();
    
private:
    
    RawData memory;                         // the actual data passed to the kernel
    Mapping mapping;                        // variable name connected to its index in memory
    
    Indices inputVariables;                 // indices of input variables
    Indices outputVariables;                // indices of output variables
    Indices targetVariables;                // indices of target variables
    Index rateVariable;
    
private: // temporary stuff, never serialized:
    
    RawData outputs;                        // holds the most recent output
    
private:
    
    std::string getKeyForVariable(const VariableKey &variableKey) const;
    
    TINYRNN_DISALLOW_COPY_AND_ASSIGN(UnrolledTrainingContext);
};

//===------------------------------------------------------------------===//
// UnrolledTrainingContext implementation
//===------------------------------------------------------------------===//

inline UnrolledTrainingContext::UnrolledTrainingContext() : rateVariable(0)
{}

inline Index UnrolledTrainingContext::allocateOrReuseVariable(Value value, const VariableKey &variableKey)
{
    const std::string &key = this->getKeyForVariable(variableKey);
    const bool variableExists = (this->mapping.find(key) != this->mapping.end());
    
    if (variableExists)
    {
        const Index variableIndex = this->mapping[key];
        this->memory[variableIndex] = value;
        return variableIndex;
    }
    else
    {
        //std::cout << this->memory.size() << " is " << key << std::endl;
        this->memory.push_back(value);
        const Index variableIndex = (this->memory.size() - 1);
        this->mapping[key] = variableIndex;
        return variableIndex;
    }
    
    return 0;
}

inline Value UnrolledTrainingContext::evaluateVariable(const VariableKey &variableKey, Value defaultValue)
{
    const std::string &key = this->getKeyForVariable(variableKey);
    const bool variableExists = (this->mapping.find(key) != this->mapping.end());
    
    if (variableExists)
    {
        const Index variableIndex = this->mapping[key];
        //std::cout << "Variable: " << key << " = " << std::to_string(this->memory[variableIndex]) << std::endl;
        return this->memory[variableIndex];
    }
    
    //std::cout << "Variable missing: " << key << ", default to " << std::to_string(defaultValue) << std::endl;
    return defaultValue;
}

inline std::string UnrolledTrainingContext::getKeyForVariable(const VariableKey &variableKey) const
{
    std::ostringstream key;
    
    std::copy(variableKey.begin(),
              variableKey.end() - 1,
              std::ostream_iterator<Id>(key, "::"));
    
    key << variableKey.back();
    
    return key.str();
}

inline void UnrolledTrainingContext::registerInputVariable(Index variableIndex)
{
    this->inputVariables.push_back(variableIndex);
}

inline void UnrolledTrainingContext::registerOutputVariable(Index variableIndex)
{
    this->outputVariables.push_back(variableIndex);
    this->outputs.resize(this->outputVariables.size());
}

inline void UnrolledTrainingContext::registerTargetVariable(Index variableIndex)
{
    this->targetVariables.push_back(variableIndex);
}

inline void UnrolledTrainingContext::registerRateVariable(Index variableIndex)
{
    this->rateVariable = variableIndex;
}

inline UnrolledTrainingContext::Indices 
UnrolledTrainingContext::getInputVariables() const
{
    return this->inputVariables;
}

inline UnrolledTrainingContext::Indices
UnrolledTrainingContext::getOutputVariables() const
{
    return this->outputVariables;
}

inline UnrolledTrainingContext::Indices
UnrolledTrainingContext::getTargetVariables() const
{
    return this->targetVariables;
}

inline Index UnrolledTrainingContext::getRateVariable() const
{
    return this->rateVariable;
}

inline UnrolledTrainingContext::RawData &UnrolledTrainingContext::getMemory()
{
    return this->memory;
}

inline UnrolledTrainingContext::RawData &UnrolledTrainingContext::getOutputs()
{
    return this->outputs;
}

inline void UnrolledTrainingContext::clear()
{
    this->memory.clear();
    this->outputs.clear();
    this->mapping.clear();
    this->inputVariables.clear();
    this->outputVariables.clear();
    this->targetVariables.clear();
    this->rateVariable = 0;
}

inline void UnrolledTrainingContext::clearMappings()
{
    this->mapping.clear();
}

//===------------------------------------------------------------------===//
// Restore neuron state
//===------------------------------------------------------------------===//

inline void UnrolledTrainingContext::restoreNeuronState(Neuron::Ptr target)
{
    const Value bias = this->evaluateVariable({target->getUuid(), Keys::Mapping::Bias}, target->bias);
    const Value state = this->evaluateVariable({target->getUuid(), Keys::Mapping::State}, target->state);
    const Value oldState = this->evaluateVariable({target->getUuid(), Keys::Mapping::OldState}, target->oldState);
    const Value activation = this->evaluateVariable({target->getUuid(), Keys::Mapping::Activation}, target->activation);
    
    target->bias = bias;
    target->state = state;
    target->oldState = oldState;
    target->activation = activation;
    
    for (auto &i : target->eligibility)
    {
        const Id &inputConnectionUuid = i.first;
        target->eligibility[inputConnectionUuid] =
        this->evaluateVariable({target->getUuid(), inputConnectionUuid, Keys::Mapping::Eligibility},
                               target->eligibility[inputConnectionUuid]);
    }
    
    for (auto &i : target->extended)
    {
        const Id &neighbourNeuronUuid = i.first;
        Neuron::EligibilityMap &map = i.second;
        
        for (auto &j : map)
        {
            const Id &inputConnectionUuid = j.first;
            
            const Value extendedTrace =
            this->evaluateVariable({target->getUuid(), neighbourNeuronUuid, inputConnectionUuid, Keys::Mapping::ExtendedTrace},
                                   target->extended[neighbourNeuronUuid][inputConnectionUuid]);
            
            target->extended[neighbourNeuronUuid][inputConnectionUuid] = extendedTrace;
        }
    }
    
    for (auto &i : target->outgoingConnections)
    {
        auto outgoingConnection = i.second;
        auto outgoingConnectionUuid = i.first;
        
        outgoingConnection->weight = this->evaluateVariable({outgoingConnectionUuid, Keys::Mapping::Weight},
                                                            outgoingConnection->weight);
        
        outgoingConnection->gain = this->evaluateVariable({outgoingConnectionUuid, Keys::Mapping::Gain},
                                                          outgoingConnection->gain);
    }
    
    if (target->isSelfConnected())
    {
        auto selfConnection = target->getSelfConnection();
        
        selfConnection->weight = this->evaluateVariable({selfConnection->getUuid(), Keys::Mapping::Weight},
                                                        selfConnection->weight);
        
        selfConnection->gain = this->evaluateVariable({selfConnection->getUuid(), Keys::Mapping::Gain},
                                                      selfConnection->gain);
    }
}

#endif  // TINYRNN_UNROLLEDTRAININGCONTEXT_H_INCLUDED
