// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class HardSigmoid : Utility.Activation
    {
        public HardSigmoid(HardSigmoidSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < 0) ? 0 : (input < 1) ? input : 1;

        protected override double Derivative(double input) => (input > 1 || input < 0) ? 0 : 1;

        public override Utility.EActivationType Type() => Utility.EActivationType.HardSigmoid;

        public override string ToString() => Type().ToString();
    }

    public sealed class HardSigmoidSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.HardSigmoid;
    }
}
