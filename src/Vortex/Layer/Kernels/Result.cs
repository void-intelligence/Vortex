// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Layer.Utility;
using Vortex.Activation.Kernels;
using Vortex.Optimizer.Kernels;
using Vortex.Regularization.Kernels;

namespace Vortex.Layer.Kernels
{
    public class ResultKernel : BaseLayerKernel
    {
        public ResultKernel(Utility.BaseLayer settings)
            : base(settings, new GradientDescentKernel(0.01))
        {
        }

        public override Matrix Forward(Matrix inputs)
        {
            // Calculate Feed Forward Operation
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

        public override ELayerType Type() => ELayerType.Result;
    }

    public class Result : BaseLayer
    {
        public Result(int neuronCount)
            : base(neuronCount, new Identity(), new None())
        {
        }

        public override ELayerType Type() => ELayerType.Result;
    }
}
