// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class SwishKernel : BaseActivationKernel
    {
        public SwishKernel(Swish settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (Exp(-input) + 1.0);

        protected override double Derivative(double input) => (1.0 + Exp(input) + input) / Pow(1.0 + Exp(input), 2.0);

        public override EActivationType Type() => EActivationType.Swish;
    }

    public sealed class Swish : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Swish;
    }
}