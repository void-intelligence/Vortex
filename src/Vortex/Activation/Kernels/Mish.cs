// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class MishKernel : BaseActivationKernel
    {
        public MishKernel(Mish settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input * Tanh(Log(1 + Exp(input)));

        protected override double Derivative(double input)
        {
            var epsilon = 2 * Exp(input) + Exp(2 * input) + 2;

            var omega = 4 * (input + 1) + 4 * Exp(2 * input) + Exp(3 * input) + Exp(input) * (4 * input + 6);

            return Exp(input) * omega / Pow(epsilon, 2);
        }

        public override EActivationType Type() => EActivationType.Mish;
    }

    public sealed class Mish : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Mish;
    }
}
