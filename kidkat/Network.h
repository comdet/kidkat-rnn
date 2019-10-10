#ifndef TINYRNN_NETWORK_H_INCLUDED
#define TINYRNN_NETWORK_H_INCLUDED

#include "Common.h"
#include "Layer.h"
#include "ScopedTimer.h"
#include "UnrolledNetwork.h"
#include "UnrolledTrainingContext.h"

class Network final
{
public:
    
    using Ptr = std::shared_ptr<Network>;
    using WeakPtr = std::weak_ptr<Network>;
    
public:
    
    Network();
    
    Network(char *networkName,
            Layer::Ptr inputLayer,
            Layer::Vector hiddenLayers,
            Layer::Ptr outputLayer);
    
    char * getName() const noexcept;
    Id getUuid() const noexcept;
    
    // Feed the input layer, process the rest and get result values from the output
    Neuron::Values feed(const Neuron::Values &input);
    
    // Back-propagation magic
    void train(Value rate, const Neuron::Values &target);
    
    // Connections
    Neuron::Connection::HashMap connectAllToAll(Network::Ptr other);
    Neuron::Connection::HashMap connectOneToOne(Network::Ptr other);
    
    // Gating
    bool gateAllIncomingConnections(Network::Ptr toNetwork, const Neuron::Connection::HashMap &connections);
    bool gateAllOutgoingConnections(Network::Ptr fromNetwork, const Neuron::Connection::HashMap &connections);
    bool gateOneToOne(Network::Ptr fromNetwork, Network::Ptr toNetwork, const Neuron::Connection::HashMap &connections);
    
public:
    
    struct Prefabs
    {
        static Network::Ptr feedForward(char *name,
                                        int inputLayerSize,
                                        const std::vector<int> &hiddenLayersSizes,
                                        int outputLayerSize);
        
        static Network::Ptr longShortTermMemory(char *name,
                                                int inputLayerSize,
                                                const std::vector<int> &hiddenLayersSizes,
                                                int outputLayerSize);
    };
    
public:
    
    UnrolledNetwork::Ptr toVM() const;
    UnrolledNetwork::Ptr toStaticVM() const;
    //void restore(UnrolledTrainingContext::Ptr context);
    
private:
    
    char * name;
    Id uuid;
    
    Layer::Ptr inputLayer;
    Layer::Vector hiddenLayers;
    Layer::Ptr outputLayer;
    
private:
    
    TINYRNN_DISALLOW_COPY_AND_ASSIGN(Network);
};

//===------------------------------------------------------------------===//
// Network implementation
//===------------------------------------------------------------------===//

inline Network::Network() :
uuid(Uuid::generateId())
{
}

inline Network::Network(char * networkName,
                 Layer::Ptr targetInputLayer,
                 Layer::Vector targetHiddenLayers,
                 Layer::Ptr targetOutputLayer) :
uuid(Uuid::generateId()),
inputLayer(targetInputLayer),
hiddenLayers(targetHiddenLayers),
outputLayer(targetOutputLayer)
{
    name = networkName;
}

inline char * Network::getName() const noexcept
{
    return this->name;
}

inline Id Network::getUuid() const noexcept
{
    return this->uuid;
}

//===------------------------------------------------------------------===//
// Core
//===------------------------------------------------------------------===//

inline Neuron::Values Network::feed(const Neuron::Values &input)
{
    this->inputLayer->feed(input);
    
    for (auto &hiddenLayer : this->hiddenLayers)
    {
        hiddenLayer->process();
    }
    
    const Neuron::Values &result = this->outputLayer->process();
    return result;
}

inline void Network::train(Value rate, const Neuron::Values &target)
{
    this->outputLayer->train(rate, target);
    
    for (size_t i = this->hiddenLayers.size(); i --> 0 ;)
    {
        this->hiddenLayers[i]->backPropagate(rate);
    }
}

//===------------------------------------------------------------------===//
// Connections
//===------------------------------------------------------------------===//

inline Neuron::Connection::HashMap Network::connectAllToAll(Network::Ptr other)
{
    return this->outputLayer->connectAllToAll(other->inputLayer);
}

inline Neuron::Connection::HashMap Network::connectOneToOne(Network::Ptr other)
{
    return this->outputLayer->connectOneToOne(other->inputLayer);
}

inline bool Network::gateAllIncomingConnections(Network::Ptr toNetwork, const Neuron::Connection::HashMap &connections)
{
    return this->outputLayer->gateAllIncomingConnections(toNetwork->inputLayer, connections);
}

inline bool Network::gateAllOutgoingConnections(Network::Ptr fromNetwork, const Neuron::Connection::HashMap &connections)
{
    return this->outputLayer->gateAllOutgoingConnections(fromNetwork->outputLayer, connections);
}

inline bool Network::gateOneToOne(Network::Ptr fromNetwork, Network::Ptr toNetwork, const Neuron::Connection::HashMap &connections)
{
    return this->outputLayer->gateOneToOne(fromNetwork->outputLayer, toNetwork->inputLayer, connections);
}

//===------------------------------------------------------------------===//
// Unrolled networks
//===------------------------------------------------------------------===//

inline UnrolledNetwork::Ptr Network::toVM() const
{
    UnrolledTrainingContext::Ptr context(new UnrolledTrainingContext());
    UnrolledNetwork::VMLayers vmLayers;
    
    {
        const ScopedTimer timer("Network::toVM");
        vmLayers.push_back(this->inputLayer->toVM(context, true, false, false));
        
        for (auto &hiddenLayer : this->hiddenLayers)
        {
            vmLayers.push_back(hiddenLayer->toVM(context, false, false, false));
        }
        
        vmLayers.push_back(this->outputLayer->toVM(context, false, true, false));
    }
    
    UnrolledNetwork::Ptr vmNetwork(new UnrolledNetwork(context, vmLayers));
    
    std::cout << "Hardcoded context memory size: " << context->getMemory().size() << std::endl;
    return vmNetwork;
}

inline UnrolledNetwork::Ptr Network::toStaticVM() const
{
    UnrolledTrainingContext::Ptr context(new UnrolledTrainingContext());
    UnrolledNetwork::VMLayers vmLayers;
    
    {
        const ScopedTimer timer("Network::toFeedOnlyVM");
        vmLayers.push_back(this->inputLayer->toVM(context, true, false, true));
        
        for (auto &hiddenLayer : this->hiddenLayers)
        {
            vmLayers.push_back(hiddenLayer->toVM(context, false, false, true));
        }
        
        vmLayers.push_back(this->outputLayer->toVM(context, false, true, true));
    }
    
    UnrolledNetwork::Ptr vmNetwork(new UnrolledNetwork(context, vmLayers));
    
    std::cout << "Hardcoded context memory size: " << context->getMemory().size() << std::endl;
    return vmNetwork;
}

//===------------------------------------------------------------------===//
// Network prefabs
//===------------------------------------------------------------------===//

inline Network::Ptr Network::Prefabs::feedForward(char *name,
                                                  int inputLayerSize,
                                                  const std::vector<int> &hiddenLayersSizes,
                                                  int outputLayerSize)
{
    Layer::Ptr inputLayer = Layer::Ptr(new Layer(inputLayerSize));
    
    std::vector<Layer::Ptr> hiddenLayers;
    Layer::Ptr prevHiddenLayer = nullptr;
    
    for (size_t i = 0; i < hiddenLayersSizes.size(); ++i)
    {
        Layer::Ptr hiddenLayer = Layer::Ptr(new Layer(hiddenLayersSizes[i]));
        
        if (i == 0)
        {
            inputLayer->connectAllToAll(hiddenLayer);
        }
        else if (prevHiddenLayer != nullptr)
        {
            prevHiddenLayer->connectAllToAll(hiddenLayer);
        }
        
        prevHiddenLayer = hiddenLayer;
        hiddenLayers.push_back(hiddenLayer);
    }
    
    Layer::Ptr outputLayer(new Layer(outputLayerSize));
    prevHiddenLayer->connectAllToAll(outputLayer);
    
    Network::Ptr network = Network::Ptr(new Network(name, inputLayer, hiddenLayers, outputLayer));
    return network;
}

inline Network::Ptr Network::Prefabs::longShortTermMemory(char *name,
                                                          int inputLayerSize,
                                                          const std::vector<int> &hiddenLayersSizes,
                                                          int outputLayerSize)
{
    Layer::Ptr inputLayer(new Layer(inputLayerSize, Neuron::Sigmoid));
    Layer::Ptr outputLayer(new Layer(outputLayerSize, Neuron::Tanh));
    
    const int numHiddenLayers = hiddenLayersSizes.size();
    Layer::Vector hiddenLayers;
    Layer::Ptr previous;
    
    for (int i = 0; i < numHiddenLayers; ++i)
    {
        const int size = hiddenLayersSizes[i];
        
        Layer::Ptr inputGate(new Layer(size, 1.0, Neuron::Sigmoid));
        Layer::Ptr forgetGate(new Layer(size, 1.0, Neuron::Sigmoid));
        Layer::Ptr memoryCell(new Layer(size, Neuron::Tanh));
        Layer::Ptr outputGate(new Layer(size, 1.0, Neuron::Sigmoid));
        
        hiddenLayers.push_back(inputGate);
        hiddenLayers.push_back(forgetGate);
        hiddenLayers.push_back(memoryCell);
        hiddenLayers.push_back(outputGate);
        
        const auto &input = inputLayer->connectAllToAll(memoryCell);
        inputLayer->connectAllToAll(inputGate);
        inputLayer->connectAllToAll(forgetGate);
        inputLayer->connectAllToAll(outputGate);
        
        Neuron::Connection::HashMap cell;
        
        if (previous != nullptr)
        {
            cell = previous->connectAllToAll(memoryCell);
            previous->connectAllToAll(inputGate);
            previous->connectAllToAll(forgetGate);
            previous->connectAllToAll(outputGate);
        }
        
        const auto &output = memoryCell->connectAllToAll(outputLayer);
        
        const auto &self = memoryCell->connectOneToOne(memoryCell);
        
        // optional
        //outputLayer->connectAllToAll(memoryCell);
        
        // optional
        //outputLayer->connectAllToAll(inputGate);
        //outputLayer->connectAllToAll(outputGate);
        //outputLayer->connectAllToAll(forgetGate);
        
        // optional
        //memoryCell->connectOneToOne(inputGate);
        memoryCell->connectAllToAll(inputGate);
        memoryCell->connectAllToAll(forgetGate);
        memoryCell->connectAllToAll(outputGate);
        
        // gates
        inputGate->gateAllIncomingConnections(memoryCell, input);
        forgetGate->gateOneToOne(memoryCell, memoryCell, self);
        outputGate->gateAllOutgoingConnections(memoryCell, output);
        
        if (previous != nullptr)
        {
            inputGate->gateAllIncomingConnections(memoryCell, cell);
        }
        
        previous = memoryCell;
    }
    
    // optional
    inputLayer->connectAllToAll(outputLayer);
    
    Network::Ptr network = Network::Ptr(new Network(name, inputLayer, hiddenLayers, outputLayer));
    return network;
}

#endif // TINYRNN_NETWORK_H_INCLUDED
