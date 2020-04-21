// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Softplus : Utility.Activation
    {
        public Softplus(SoftplusSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => System.Math.Log(1 + System.Math.Exp(input));

        protected override double Derivative(double input) => System.Math.Exp(input) / (1 + System.Math.Exp(input));

        public override Utility.EActivationType Type() => Utility.EActivationType.Softplus;

        public override string ToString() => Type().ToString();
    }

    public sealed class SoftplusSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Softplus;
    }
}
