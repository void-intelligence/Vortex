// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Layer.Utility;

namespace Vortex.Layer.Kernels
{
    public class Dense : BaseLayer
    {
#nullable enable
        public Dense(int neuronCount, IActivation? activation = null, IRegularization? regularization = null,
            IInitializer? initializer = null, IMutation? mutation = null, IOptimizer? optimizer = null)
            : base(neuronCount, activation, regularization, initializer, mutation, optimizer)
        {
        }
#nullable disable

        public override Matrix Forward(Matrix inputs)
        {
            if (MutationFunction.Type() != EMutationType.NoMutation) Params["W"].InMap(MutationFunction.Mutate);

            // Calculate Regularization Value On W
            RegularizationValue = (float) RegularizationFunction.CalculateNorm(Params["W"]);

            // Calculate Feed Forward Operation
            Params["X"] = inputs;
            Params["Z"] = Params["W"] * Params["X"] + Params["B"];
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

        public override ELayerType Type()
        {
            return ELayerType.Dense;
        }
    }
}
