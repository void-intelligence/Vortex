// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class LoggyKernel : BaseActivationKernel
    {
        public LoggyKernel(Loggy settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Tanh(input / 2.0);

        protected override double Derivative(double input) => 1.0 / (Cosh(input) + 1.0);

        public override EActivationType Type() => EActivationType.Loggy;
    }

    public sealed class Loggy : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Loggy;
    }
}