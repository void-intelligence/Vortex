// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class ReluKernel : BaseActivationKernel
    {
        public ReluKernel(Relu settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Max(input, 0);

        protected override double Derivative(double input) => (input > 0) ? 1 : 0;

        public override EActivationType Type() => EActivationType.Relu;
    }

    public sealed class Relu : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Relu;
    }
}
