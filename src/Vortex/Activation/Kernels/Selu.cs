// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class SeluKernel : Utility.BaseActivationKernel
    {
        public SeluKernel(Selu settings = null) : base(settings) { }

        public const double Alpha = 1.6732632423543772848170429916717;

        public const double Lambda = 1.0507009873554804934193349852946;

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input > 0) ? input : Alpha * (Exp(input) - 1);

        protected override double Derivative(double input) => (input > 0) ? Lambda : Lambda * Alpha * Exp(input);

        public override EActivationType Type() => EActivationType.Selu;
    }

    public sealed class Selu : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Selu;
    }
}
