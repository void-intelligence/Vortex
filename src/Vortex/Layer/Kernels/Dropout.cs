// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Activation.Utility;
using Vortex.Regularization.Utility;
using Vortex.Optimizer.Utility;
using Vortex.Initializer.Utility;
using Vortex.Mutation.Utility;
using Vortex.Layer.Utility;

namespace Vortex.Layer.Kernels
{
    public class Dropout : BaseLayer
    {
        public float DropoutChance { get; set; }
#nullable enable
        public Dropout(int neuronCount, float dropoutChance, BaseActivation activation, BaseRegularization? regularization = null,
            BaseInitializer? initializer = null, BaseMutation? mutation = null, BaseOptimizer? optimizer = null)
            : base(neuronCount, activation, regularization, initializer, mutation, optimizer)
        {
            DropoutChance = dropoutChance;
        }
#nullable disable

        public override Matrix Forward(Matrix inputs)
        {
            if (MutationFunction.Type() != EMutationType.NoMutation) Params["W"].InMap(MutationFunction.Mutate);

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

        public override ELayerType Type()
        {
            return ELayerType.Dropout;
        }
    }
}
