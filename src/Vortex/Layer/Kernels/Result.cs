// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Activation.Kernels;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Layer.Utility;

namespace Vortex.Layer.Kernels
{
    public class Result : BaseLayer
    {
#nullable enable
        public Result(int neuronCount, BaseRegularization? regularization = null,
            BaseInitializer? initializer = null, BaseMutation? mutation = null, BaseOptimizer? optimizer = null)
            : base(neuronCount, new Identity(), regularization, initializer, mutation, optimizer)
        {
        }
#nullable disable

        public override Matrix Forward(Matrix inputs)
        {
            // The result layer just holds the values of the input
            Params["X"] = inputs;
            return Params["X"];
        }

        public override Matrix Backward(Matrix dA)
        {
            throw new InvalidOperationException("Backprop should not be called on the result layer!");
        }

        public override void Optimize()
        {
        }

        public override ELayerType Type()
        {
            return ELayerType.Result;
        }
    }
}
