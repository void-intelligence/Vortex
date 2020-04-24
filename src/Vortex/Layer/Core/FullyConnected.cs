// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Vortex.Layer.Utility;
using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;

namespace Vortex.Layer
{
    public class FullyConnected : BaseLayer
    {
        public FullyConnected(LayerSettings settings) 
            : base(settings)
        {
        }

        public override Matrix Forward(Matrix inputs)
        {
            Params["X"] = inputs;
            Params["Z"] = (Params["W"].T() * Params["X"]) + Params["B"];
            Params["A"] = ActivationFunction.Forward(Params["Z"]);
            return Params["A"];
        }

        public override Matrix Backward(Matrix dA, Matrix prevA)
        {
            Grads["DA"] = dA;
            Grads["G'"] = ActivationFunction.Backward(Params["Z"]);
            Grads["DZ"] = Grads["DA"].Hadamard(Grads["G'"]);
            Grads["DW"] = Grads["DZ"] * prevA;
            Grads["DB"] = Grads["DZ"];
            Grads["DA-1"] = Params["W"].T() * Grads["DZ"];
            return Grads["DA-1"];
        }

        public override void Optimize(Matrix wGrad, Matrix bGrad)
        {
            Params["W"] = Params["W"] - wGrad;
            Params["B"] = Params["B"] - bGrad;
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
