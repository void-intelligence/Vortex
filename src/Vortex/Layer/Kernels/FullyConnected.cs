// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Layer.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;

namespace Vortex.Layer.Kernels
{
    public class FullyConnectedKernel : BaseLayerKernel
    {
        public FullyConnectedKernel(BaseLayer settings) 
            : base(settings)
        {
        }

        public override Matrix Forward(Matrix inputs)
        {
            Params["W"].InMap(MutationFunction.Mutate);

            // Calculate Regularization Value On W and B
            RegularizationValue = (float) RegularizationFunction.CalculateNorm(Params["W"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = (Params["W"] * Params["X"]) + Params["B"];
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

        public override ELayerType Type() => ELayerType.FullyConnected;
    }

    public class FullyConnected : BaseLayer
    {
#nullable enable
        public FullyConnected(int neuronCount, BaseActivation activation, BaseRegularization? regularization = null, BaseInitializer? initializer = null, BaseMutation? mutation = null)
            : base(neuronCount, activation, regularization, initializer, mutation)
        {
        }
#nullable disable

        public override ELayerType Type() => ELayerType.FullyConnected;
    }
}
