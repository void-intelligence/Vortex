// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class BipolarSigmoid : Utility.Activation
    {
        public BipolarSigmoid(BipolarSigmoidSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => -1 + 2 / (1 + System.Math.Exp(-input));

        protected override double Derivative(double input) => 0.5 * (1 + (-1 + 2 / (1 + System.Math.Exp(-input)))) * (1 - (-1 + 2 / (1 + System.Math.Exp(-input))));

        public override Utility.EActivationType Type() => Utility.EActivationType.BipolarSigmoid;

        public override string ToString() => Type().ToString();
    }

    public sealed class BipolarSigmoidSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.BipolarSigmoid;
    }
}
