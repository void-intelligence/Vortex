// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class ReLU : Utility.BaseActivation
    {
        public ReLU(ReLUSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => System.Math.Max(input, 0);

        protected override double Derivative(double input) => (input > 0) ? 1 : 0;

        public override Utility.EActivationType Type() => Utility.EActivationType.ReLU;

        public override string ToString() => Type().ToString();
    }

    public sealed class ReLUSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.ReLU;
    }
}
