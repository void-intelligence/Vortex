// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using System.Collections.Generic;
using System.Text;
using Nomad.Core;
using Vortex.Activation.Utility;
using Vortex.Initializer.Utility;
using Vortex.Layer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Pooler.Kernels;
using Vortex.Pooler.Utility;
using Vortex.Regularization.Utility;

namespace Vortex.Layer.Kernels
{
    public class Convolutional : BaseLayer
    {
#nullable enable

        public IPooler PoolerFunction { get; }

        public Convolutional(int inputFeatures, IPooler? pooler = null, IActivation? activation = null, IRegularization? regularization = null, IInitializer? initializer = null, IMutation? mutation = null, IOptimizer? optimizer = null) : base(inputFeatures, activation, regularization, initializer, mutation, optimizer)
        {
            PoolerFunction = pooler ?? new Average();
        }
#nullable disable

        public override Matrix Forward(Matrix inputs)
        {
            if (MutationFunction.Type() != EMutationType.NoMutation) Params["W"].InMap(MutationFunction.Mutate);

            // Calculate Regularization Value On W
            RegularizationValue = (float)RegularizationFunction.CalculateNorm(Params["W"]);
            
            // Calculate Convolution Operation
            Params["X"] = inputs.Reshape((int)Math.Sqrt(NeuronCount), (int)Math.Sqrt(NeuronCount));
            if (!(((BasePooler) PoolerFunction).PadSizeCol == 0 || ((BasePooler) PoolerFunction).PadSizeRow == 0))
            {
                Params["PAD"] = PoolerFunction.Pad(Params["X"], Params["W"]);
                Params["Z"] = PoolerFunction.Pool(Params["PAD"], Params["W"]);
            }
            else
            {
                Params["Z"] = PoolerFunction.Pool(Params["X"], Params["W"]);
            }

            Params["Z"].InFlatten();
            Params["A"] = ActivationFunction.Forward(Params["Z"]);
            return Params["A"];
        }

        public override Matrix Backward(Matrix dA)
        {
            Grads["DA"] = dA;
            Grads["G'"] = ActivationFunction.Backward(Params["Z"]);
            Grads["DZ"] = Grads["DA"].Hadamard(Grads["G'"]);
            Grads["DW"] = Grads["DZ"] * Params["X"].T();
            Grads["DB"] = Grads["DZ"];
            return Params["W"].T() * Grads["DZ"];
        }

        public override ELayerType Type()
        {
            return ELayerType.Convolutional;
        }
    }
}
