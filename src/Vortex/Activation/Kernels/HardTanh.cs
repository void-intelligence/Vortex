// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class HardTanhKernel : BaseActivationKernel
    {
        public HardTanhKernel(HardTanh settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < -1) ? -1 : (input > 1) ? 1 : input;

        protected override double Derivative(double input) => (input < -1) ? 0 : (input > 1) ? 0 : 1;

        public override EActivationType Type() => EActivationType.HardTanh;
    }

    public sealed class HardTanh : BaseActivation
    {
        public override EActivationType Type() => EActivationType.HardTanh;
    }
}
