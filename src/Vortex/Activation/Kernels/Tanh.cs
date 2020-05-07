// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class TanhKernel : BaseActivationKernel
    {
        public TanhKernel(Tanh settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (Exp(input) - Exp(-input)) / (Exp(input) + Exp(-input));
        
        protected override double Derivative(double input) => 1 - Pow((Exp(input) - Exp(-input)) / (Exp(input) + Exp(-input)), 2);

        public override EActivationType Type() => EActivationType.Tanh;
    }

    public sealed class Tanh : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Tanh;
    }
}
