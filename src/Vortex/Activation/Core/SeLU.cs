// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class SeLU : Utility.BaseActivation
    {
        public SeLU(SeLUSettings settings = null) : base(settings) { }

        public const double Alpha = 1.6732632423543772848170429916717;

        public const double Lambda = 1.0507009873554804934193349852946;

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input > 0) ? input : Alpha * (System.Math.Exp(input) - 1);

        protected override double Derivative(double input) => (input > 0) ? Lambda : Lambda * Alpha * System.Math.Exp(input);

        public override Utility.EActivationType Type() => Utility.EActivationType.SeLU;
    }

    public sealed class SeLUSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.SeLU;
    }
}
