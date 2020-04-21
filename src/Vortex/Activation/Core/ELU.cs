// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class ELU : Utility.BaseActivation
    {
        public double Alpha { get; set; }

        public ELU(ELUSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input >= 0) ? input : Alpha * (System.Math.Exp(input) - 1);

        protected override double Derivative(double input) => (input >= 0) ? 1 : Alpha * System.Math.Exp(input);

        public override Utility.EActivationType Type() => Utility.EActivationType.ELU;

        public override string ToString() => Type().ToString();
    }

    public sealed class ELUSettings : Utility.ActivationSettings
    {
        public ELUSettings(double alpha = 0.01)
        {
            Alpha = alpha;
        }

        public double Alpha { get; private set; }

        public override Utility.EActivationType Type() => Utility.EActivationType.ELU;
    }
}
