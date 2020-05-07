// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class SoftsignKernel : BaseActivationKernel
    {
        public SoftsignKernel(Softsign settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input / (1 + Abs(input));

        protected override double Derivative(double input) => input / Pow((1 + Abs(input)), 2);   

        public override EActivationType Type() => EActivationType.Softsign;
    }

    public sealed class Softsign : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Softsign;
    }
}
