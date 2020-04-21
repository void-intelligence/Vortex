// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Softsign : Utility.Activation
    {
        public Softsign(SoftsignSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input / (1 + System.Math.Abs(input));

        protected override double Derivative(double input) => input / System.Math.Pow((1 + System.Math.Abs(input)), 2);   

        public override Utility.EActivationType Type() => Utility.EActivationType.Softsign;

        public override string ToString() => Type().ToString();
    }

    public sealed class SoftsignSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Softsign;
    }
}
