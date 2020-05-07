// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class IdentityKernel : BaseActivationKernel
    {
        public IdentityKernel(Identity settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input;

        protected override double Derivative(double input) => 1;

        public override EActivationType Type() => EActivationType.Identity;
    }

    public sealed class Identity : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Identity;
    }
}
