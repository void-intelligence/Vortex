﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Layer.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Initializers.Utility;

namespace Vortex.Layer.Kernels
{
    public class DropoutKernel : BaseLayerKernel
    {
        public float DropoutChance;

        public DropoutKernel(Dropout settings, BaseOptimizerKernel optimizer)
            : base(settings, optimizer)
        {
            DropoutChance = settings.DropoutChance;
        }

        public override Matrix Forward(Matrix inputs)
        {
            // Calculate Regularization Value On W and B
            RegularizationValue = (float) RegularizationFunction.CalculateNorm(Params["W"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = Params["W"] * Params["X"] + Params["B"];
            Params["A"] = ActivationFunction.Forward(Params["Z"]);

            // Dropout
            Params["A"].InDropout(DropoutChance);
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

        public override void Optimize()
        {
            Matrix deltaW = OptimizerFunction.CalculateDeltaW(Params["W"], Grads["DW"]);
            Matrix deltaB = OptimizerFunction.CalculateDeltaB(Params["B"], Grads["DB"]);

            Params["W"] -= deltaW;
            Params["B"] -= deltaB;
        }

        public override ELayerType Type() => ELayerType.Dropout;
    }

    public class Dropout : BaseLayer
    {
        public float DropoutChance { get; set; }

        public Dropout(int neuronCount, BaseActivation activation, BaseRegularization regularization, BaseInitializer initializer, float dropoutChance = 0.5f)
            : base(neuronCount, activation, regularization, initializer)
        {
            DropoutChance = dropoutChance;
        }

        public override ELayerType Type() => ELayerType.Dropout;
    }
}
