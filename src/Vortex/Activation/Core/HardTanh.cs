// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class HardTanh : Utility.BaseActivation
    {
        public HardTanh(HardTanhSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < -1) ? -1 : (input > 1) ? 1 : input;

        protected override double Derivative(double input) => (input < -1) ? 0 : (input > 1) ? 0 : 1;

        public override Utility.EActivationType Type() => Utility.EActivationType.HardTanh;

        public override string ToString() => Type().ToString();
    }

    public sealed class HardTanhSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.HardTanh;
    }
}
