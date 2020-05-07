// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class BinaryStepKernel : BaseActivationKernel
    {
        public BinaryStepKernel(BinaryStep settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < 0) ? 0 : 1;

        protected override double Derivative(double input) => 0;

        public override EActivationType Type() => EActivationType.BinaryStep;
    }

    public sealed class BinaryStep : BaseActivation
    {
        public override EActivationType Type() => EActivationType.BinaryStep;
    }
}
