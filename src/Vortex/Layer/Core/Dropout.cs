// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Layer.Utility;
using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Layer
{
    public class Dropout : BaseLayer
    {
        public float DropoutChance;

        public Dropout(DropoutSettings settings, BaseOptimizer optimizer)
            : base(settings, optimizer)
        {
            DropoutChance = settings.DropoutChance;
        }

        public override Matrix Forward(Matrix inputs)
        {
            // Calculate Regularization Value On W and B
            RegularizationValue = (float)RegularizationFunction.CalculateNorm(Params["W"]) + (float)RegularizationFunction.CalculateNorm(Params["B"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = (Params["W"].T() * Params["X"]) + Params["B"];
            Params["A"] = ActivationFunction.Forward(Params["Z"]);
            
            // Dropout
            Params["A"].InDropout(DropoutChance);
            return Params["A"];
        }

        public override Matrix Backward(Matrix dA)
        {
            // Note: Params["A-1"] will be set in Network.Backward() function
            // Helper Grads
            Grads["DA"] = dA;
            Grads["G'"] = ActivationFunction.Backward(Params["Z"]);
            Grads["DZ"] = Grads["DA"].Hadamard(Grads["G'"]);

            // The Intended Grads
            Grads["DW"] = Params["A-1"] * Grads["DZ"].T();
            Grads["DB"] = Grads["DZ"];
            Grads["DA-1"] = Params["W"] * Grads["DZ"];
            return Grads["DA-1"];
        }

        public override void Optimize()
        {
            Matrix deltaW = OptimizerFunction.CalculateDeltaW(Params["W"], Grads["DW"]);
            Matrix deltaB = OptimizerFunction.CalculateDeltaB(Params["B"], Grads["DB"]);

            Params["W"] = Params["W"] - deltaW;
            Params["B"] = Params["B"] - deltaB;
        }

        public override ELayerType Type() => ELayerType.Dropout;
    }

    public class DropoutSettings : LayerSettings
    {
        public float DropoutChance { get; set; }

        public DropoutSettings(int neuronCount, ActivationSettings activation, RegularizationSettings regularization, float dropoutChance = 0.5f)
            : base(neuronCount, activation, regularization)
        {
            DropoutChance = dropoutChance;
        }

        public override ELayerType Type() => ELayerType.Dropout;
    }
}
