// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Activation;
using Vortex.Activation.Utility;
using Vortex.Regularization;
using Vortex.Regularization.Utility;
using Nomad.Matrix;
using System.Collections.Generic;

namespace Vortex.Layer.Utility
{
    /// <summary>
    /// Base of all Layer classes
    /// </summary>
    public abstract class BaseLayer
    {
        public LayerSettings Settings { get; private set; }
        public int NeuronCount { get; private set; }
        public BaseActivation ActivationFunction { get; private set; }
        public BaseRegularization RegularizationFunction { get; private set; }
        public Dictionary<string, Matrix> Params { get; private set; }
        public Dictionary<string, Matrix> Grads { get; private set; }

        public BaseLayer(LayerSettings layerSettings)
        {
            Params = new Dictionary<string, Matrix>();

            Grads = new Dictionary<string, Matrix>();

            Settings = layerSettings;

            NeuronCount = Settings.NeuronCount;

            // Activation Setup
            // Regularization Setup
            ActivationFunction = (Settings.ActivationFunction.Type()) switch
            {
                EActivationType.Invalid => null,
                EActivationType.Arctan => new Arctan(),
                EActivationType.BinaryStep => new BinaryStep(),
                EActivationType.BipolarSigmoid => new BipolarSigmoid(),
                EActivationType.ELU => new ELU((ELUSettings)Settings.ActivationFunction),
                EActivationType.HardSigmoid=> new HardSigmoid(),
                EActivationType.HardTanh=> new HardTanh(),
                EActivationType.Identity=> new Identity(),
                EActivationType.Logit=> new Logit(),
                EActivationType.LReLU=> new LReLU((LReLUSettings)Settings.ActivationFunction),
                EActivationType.Mish=> new Mish(),
                EActivationType.ReLU => new ReLU(),
                EActivationType.SeLU => new SeLU(),
                EActivationType.Sigmoid => new Sigmoid(),
                EActivationType.Softmax => new Softmax(),
                EActivationType.Softplus => new Softplus(),
                EActivationType.Softsign => new Softsign(),
                EActivationType.Tanh => new Tanh(),
                _ => throw new ArgumentException("Activation Type Invalid."),
            };

            // Regularization Setup
            RegularizationFunction = (Settings.RegularizationFunction.Type()) switch
            {
                ERegularizationType.Invalid => null,
                ERegularizationType.None => new Regularization.None(),
                ERegularizationType.L1 => new Regularization.L1((L1Settings)Settings.RegularizationFunction),
                ERegularizationType.L2 => new Regularization.L2((L2Settings)Settings.RegularizationFunction),
                _ => throw new ArgumentException("Regularization Type Invalid."),
            };
        }

        // All Layer Forward Calculations
        public abstract Matrix Forward(Matrix inputs);

        // All Layer Backward Calculations
        public abstract Matrix Backward(Matrix dA, Matrix prevA);

        // All Layer Backward Calculations
        public abstract void Optimize(Matrix wGrad, Matrix bGrad);
    }

    public class LayerSettings
    {
        public int NeuronCount { get; private set; }
        public ActivationSettings ActivationFunction { get; private set; }
        public RegularizationSettings RegularizationFunction { get; private set; }

        public LayerSettings(int neuronCount, ActivationSettings activation, RegularizationSettings regularization) 
        {
            NeuronCount = neuronCount;
            ActivationFunction = activation;
            RegularizationFunction = regularization;
        }
    }
}
