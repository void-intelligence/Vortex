using System;
using System.Collections.Generic;

using Nomad.Matrix;

using Vortex.Cost;
using Vortex.Cost.Utility;

using Vortex.Layer;
using Vortex.Layer.Utility;

using Vortex.Optimizer;
using Vortex.Optimizer.Utility;

using Vortex.Activation;
using Vortex.Activation.Utility;

using Vortex.Regularization;
using Vortex.Regularization.Utility;

namespace Vortex.Network
{
    public class Network
    {

        public List<BaseLayer> Layers { get; private set; }

        public bool IsLocked { get; private set; }

        public Network()
        {
            IsLocked = false;
            Layers = new List<BaseLayer>();
        }

        public void CreateLayer(int neuronCount, ActivationSettings activation, RegularizationSettings regularization)
        {
            if (IsLocked)
            {
                throw new InvalidOperationException("Network is Locked.");
            }

            // To-Do: Add More Layer Types
            Layers.Add(new FullyConnected(new LayerSettings(neuronCount, activation, regularization)));
        }

        public void InitNetwork()
        {
            if (IsLocked)
            {
                throw new InvalidOperationException("Network is Locked.");
            }
            IsLocked = true;

            // Initialize All Layers, Their Ws and Bs
            for (int i = 0; i < Layers.Count - 1; i++)
            {
                // Weights
                Layers[i].Params["W"] = new Matrix(Layers[i].NeuronCount, Layers[i + 1].NeuronCount);
                Layers[i].Params["W"].InRandomize();

                // Inputs
                Layers[i].Params["X"] = new Matrix(Layers[i].NeuronCount, 1);

                // Biases
                Layers[i].Params["B"] = new Matrix(Layers[i].NeuronCount, 1);
                Layers[i].Params["B"].InRandomize();
            }
        }
    }
}
