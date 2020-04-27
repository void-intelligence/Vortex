// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Identity : Utility.BaseActivation
    {
        public Identity(IdentitySettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input;

        protected override double Derivative(double input) => 1;

        public override Utility.EActivationType Type() => Utility.EActivationType.Identity;
    }

    public sealed class IdentitySettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Identity;
    }
}
