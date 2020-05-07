// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Vortex.Layer.Utility;
using Nomad.Matrix;
using Vortex.Activation;
using Vortex.Activation.Utility;
using Vortex.Optimizer;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Regularization;

namespace Vortex.Layer
{
    public class Result : BaseLayer
    {
        public Result(LayerSettings settings)
            : base(settings, new GradientDescent(0.01))
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

    public class ResultSettings : LayerSettings
    {
        public ResultSettings(int neuronCount)
            : base(neuronCount, new IdentitySettings(), new NoneSettings())
        {
        }

        public override ELayerType Type() => ELayerType.Result;
    }
}
