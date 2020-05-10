﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Layer.Utility;
using Vortex.Activation.Kernels;
using Vortex.Regularization.Kernels;
using Vortex.Initializer.Kernels;
using Vortex.Mutation.Kernels;

namespace Vortex.Layer.Kernels
{
    public class ResultKernel : BaseLayerKernel
    {
        public ResultKernel(BaseLayer settings)
            : base(settings)
        {
        }

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

    public class Result : BaseLayer
    {
        public Result(int neuronCount)
            : base(neuronCount, new Identity(), new None(), new One(0,0), new NoMutation())
        {
        }

        public override ELayerType Type()
        {
            return ELayerType.Result;
        }
    }
}
