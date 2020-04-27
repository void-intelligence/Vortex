// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Tanh : Utility.BaseActivation
    {
        public Tanh(TanhSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (System.Math.Exp(input) - System.Math.Exp(-input)) / (System.Math.Exp(input) + System.Math.Exp(-input));
        
        protected override double Derivative(double input) => 1 - System.Math.Pow((System.Math.Exp(input) - System.Math.Exp(-input)) / (System.Math.Exp(input) + System.Math.Exp(-input)), 2);

        public override Utility.EActivationType Type() => Utility.EActivationType.Tanh;
    }

    public sealed class TanhSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Tanh;
    }
}
