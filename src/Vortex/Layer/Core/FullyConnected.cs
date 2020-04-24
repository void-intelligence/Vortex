// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Layer.Utility;
using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Layer
{
    public class FullyConnected : BaseLayer
    {
        public FullyConnected(LayerSettings settings, BaseOptimizer optimizer) 
            : base(settings, optimizer)
        {
        }

        public override Matrix Forward(Matrix inputs)
        {
            // Calculate Regularization Value
            RegularizationValue = (float)RegularizationFunction.CalculateNorm(Params["W"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = (Params["W"].T() * Params["X"]) + Params["B"];
            Params["A"] = ActivationFunction.Forward(Params["Z"]);
            return Params["A"];
        }

        public override Matrix Backward(Matrix dA)
        {
            // Note: Params["A-1"] will be set in Network.Backward() function
            Grads["DA"] = dA;
            Grads["G'"] = ActivationFunction.Backward(Params["Z"]);
            Grads["DZ"] = Grads["DA"].Hadamard(Grads["G'"]);
            Grads["DW"] = Grads["DZ"] * Params["A-1"];
            Grads["DB"] = Grads["DZ"];
            Grads["DA-1"] = Params["W"].T() * Grads["DZ"];
            return Grads["DA-1"];
        }

        public override void Optimize()
        {
            Matrix deltaW = OptimizerFunction.CalculateDeltaW(Params["W"], Grads["DW"]);
            Matrix deltaB = OptimizerFunction.CalculateDeltaB(Params["B"], Grads["DB"]);

            Params["W"] = Params["W"] - deltaW;
            Params["B"] = Params["B"] - deltaB;
        }
    }

    public class FullyConnectedSettings: LayerSettings
    {
        public FullyConnectedSettings(int neuronCount, ActivationSettings activation, RegularizationSettings regularization)
            : base(neuronCount, activation, regularization)
        {
        }
    }
}
