// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System.Data;
using Vortex.Layer.Utility;
using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Layer
{
    public class Output : BaseLayer
    {
        public Output(LayerSettings settings, BaseOptimizer optimizer) 
            : base(settings, optimizer)
        {
        }

        public override Matrix Forward(Matrix inputs)
        {
            // Calculate Regularization Value On W and B
            RegularizationValue = (float)RegularizationFunction.CalculateNorm(Params["W"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = Params["W"] * Params["X"] + Params["B"];
            Params["A"] = ActivationFunction.Forward(Params["Z"]);
            return Params["A"];
        }

        public override Matrix Backward(Matrix error)
        {
            Grads["DA"] = error;
            Grads["G'"] = ActivationFunction.Backward(Params["Z"]);
            Grads["DZ"] = Grads["DA"].Hadamard(Grads["G'"]);
            Grads["DW"] = (Grads["DZ"] * Params["X"].T());
            Grads["DB"] = Grads["DZ"];
            return Grads["DZ"];
        }

        public override void Optimize()
        {
            Matrix deltaW = OptimizerFunction.CalculateDeltaW(Params["W"], Grads["DW"]);
            Matrix deltaB = OptimizerFunction.CalculateDeltaB(Params["B"], Grads["DB"]);

            Params["W"] -= deltaW;
            Params["B"] -= deltaB;
        }

        public override ELayerType Type() => ELayerType.Output;
    }

    public class OutputSettings: LayerSettings
    {
        public OutputSettings(int neuronCount, ActivationSettings activation, RegularizationSettings regularization)
            : base(neuronCount, activation, regularization)
        {
        }

        public override ELayerType Type() => ELayerType.Output;
    }
}
