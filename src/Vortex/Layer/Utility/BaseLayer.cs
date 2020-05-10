// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Activation.Kernels;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Nomad.Matrix;
using System.Collections.Generic;
using Vortex.Optimizer.Utility;
using Vortex.Regularization.Kernels;
using Vortex.Initializer.Kernels;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Mutation.Kernels;
using Vortex.Optimizer.Kernels;

namespace Vortex.Layer.Utility
{
    /// <summary>
    /// Base of all Layer classes
    /// </summary>
    public abstract class BaseLayer
    {
        public float RegularizationValue { get; protected set; }
        public int NeuronCount { get; }


        public BaseActivation ActivationFunction { get; }
        public BaseRegularization RegularizationFunction { get; }
        public BaseOptimizer OptimizerFunction { get; set; }
        public BaseInitializer InitializerFunction { get; }
        public BaseMutation MutationFunction { get; }


        public Dictionary<string, Matrix> Params { get; }
        public Dictionary<string, Matrix> Grads { get; }

#nullable enable
        protected BaseLayer(int neuronCount, BaseActivation activation, BaseRegularization? regularization = null,
            BaseInitializer? initializer = null, BaseMutation? mutation = null, BaseOptimizer? optimizer = null)
        {
            RegularizationValue = 0.0f;
            NeuronCount = neuronCount;

            ActivationFunction = activation;
            RegularizationFunction = regularization ?? new None();
            InitializerFunction = initializer ?? new Auto();
            MutationFunction = mutation ?? new NoMutation();
            OptimizerFunction = optimizer ?? new DefaultOptimizer();

            Params = new Dictionary<string, Matrix>();
            Grads = new Dictionary<string, Matrix>();
        }
#nullable disable

        // All Layer Forward Calculations
        public abstract Matrix Forward(Matrix inputs);

        // All Layer Backward Calculations
        public abstract Matrix Backward(Matrix dA);

        // All Layer Optimization Calculations
        public virtual void Optimize()
        {
            var deltaW = OptimizerFunction.CalculateDelta(Params["W"], Grads["DW"]);
            var deltaB = OptimizerFunction.CalculateDelta(Params["B"], Grads["DB"]);

            Params["W"] -= deltaW;
            Params["B"] -= deltaB;
        }

        public abstract ELayerType Type();
    }
}
