// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class HardSigmoidKernel : BaseActivationKernel
    {
        public HardSigmoidKernel(HardSigmoid settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < 0) ? 0 : (input < 1) ? input : 1;

        protected override double Derivative(double input) => (input > 1 || input < 0) ? 0 : 1;

        public override EActivationType Type() => EActivationType.HardSigmoid;
    }

    public sealed class HardSigmoid : BaseActivation
    {
        public override EActivationType Type() => EActivationType.HardSigmoid;
    }
}
