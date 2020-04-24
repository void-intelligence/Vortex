// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Activation;
using Vortex.Activation.Utility;
using Vortex.Regularization;
using Vortex.Regularization.Utility;
using Nomad.Matrix;
using System.Collections.Generic;
using Vortex.Optimizer.Utility;

namespace Vortex.Layer.Utility
{
    /// <summary>
    /// Base of all Layer classes
    /// </summary>
    public abstract class BaseLayer
    {
        public float RegularizationValue { get; protected set; }
        public LayerSettings Settings { get; private set; }
        public int NeuronCount { get; private set; }
        public BaseActivation ActivationFunction { get; private set; }
        public BaseRegularization RegularizationFunction { get; private set; }
        public BaseOptimizer OptimizerFunction { get; private set; }
        public Dictionary<string, Matrix> Params { get; private set; }
        public Dictionary<string, Matrix> Grads { get; private set; }

        protected BaseLayer(LayerSettings layerSettings, BaseOptimizer optimizer)
        {
            RegularizationValue = 0.0f;

            OptimizerFunction = optimizer;

            Params = new Dictionary<string, Matrix>();

            Grads = new Dictionary<string, Matrix>();

            Settings = layerSettings;

            NeuronCount = Settings.NeuronCount;

            // Activation Setup
            ActivationFunction = (Settings.ActivationFunctionSettings.Type()) switch
            {
                EActivationType.Arctan => new Arctan(),
                EActivationType.BinaryStep => new BinaryStep(),
                EActivationType.BipolarSigmoid => new BipolarSigmoid(),
                EActivationType.ELU => new ELU((ELUSettings)Settings.ActivationFunctionSettings),
                EActivationType.HardSigmoid=> new HardSigmoid(),
                EActivationType.HardTanh=> new HardTanh(),
                EActivationType.Identity=> new Identity(),
                EActivationType.Logit=> new Logit(),
                EActivationType.LReLU=> new LReLU((LReLUSettings)Settings.ActivationFunctionSettings),
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
            RegularizationFunction = (Settings.RegularizationFunctionSettings.Type()) switch
            {
                ERegularizationType.None => new Regularization.None((NoneSettings)Settings.RegularizationFunctionSettings),
                ERegularizationType.L1 => new Regularization.L1((L1Settings)Settings.RegularizationFunctionSettings),
                ERegularizationType.L2 => new Regularization.L2((L2Settings)Settings.RegularizationFunctionSettings),
                _ => throw new ArgumentException("Regularization Type Invalid."),
            };
        }

        // All Layer Forward Calculations
        public abstract Matrix Forward(Matrix inputs);

        // All Layer Backward Calculations
        public abstract Matrix Backward(Matrix dA);

        // All Layer Optimization Calculations
        public abstract void Optimize();

        public abstract ELayerType Type();
    }

    public abstract class LayerSettings
    {
        public int NeuronCount { get; private set; }
        public ActivationSettings ActivationFunctionSettings { get; private set; }
        public RegularizationSettings RegularizationFunctionSettings { get; private set; }

        protected LayerSettings(int neuronCount, ActivationSettings activationSettings, RegularizationSettings regularizationSettings) 
        {
            NeuronCount = neuronCount;
            ActivationFunctionSettings = activationSettings;
            RegularizationFunctionSettings = regularizationSettings;
        }

        public abstract ELayerType Type();
    }
}
