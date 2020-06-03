// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Nomad.Core;
using System.Collections.Generic;
using Vortex.Activation.Kernels;
using Vortex.Optimizer.Utility;
using Vortex.Regularization.Kernels;
using Vortex.Initializer.Kernels;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Mutation.Kernels;

namespace Vortex.Layer.Utility 
{
    /// <summary>
    /// Base of all Layer classes
    /// </summary>
    public abstract class BaseLayer : ILayer
    {
        public float RegularizationValue { get; protected set; }
        public int NeuronCount { get; }


        public IActivation ActivationFunction { get; }
        public IRegularization RegularizationFunction { get; }
        public IOptimizer OptimizerFunction { get; set; }
        public IInitializer InitializerFunction { get; }
        public IMutation MutationFunction { get; }


        public Dictionary<string, Matrix> Params { get; }
        public Dictionary<string, Matrix> Grads { get; }

#nullable enable
        protected BaseLayer(int neuronCount, IActivation? activation = null, IRegularization? regularization = null,
            IInitializer? initializer = null, IMutation? mutation = null, IOptimizer? optimizer = null)
        {
            RegularizationValue = 0.0f;
            NeuronCount = neuronCount;

            ActivationFunction = activation ?? new Identity();
            RegularizationFunction = regularization ?? new None();
            InitializerFunction = initializer ?? new Auto();
            MutationFunction = mutation ?? new NoMutation();
            OptimizerFunction = optimizer ?? new DefaultOptimizer();

            Params = new Dictionary<string, Matrix>();
            Grads = new Dictionary<string, Matrix>();
        }
#nullable disable

        public abstract Matrix Forward(Matrix inputs);
        public abstract Matrix Backward(Matrix dA);

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
