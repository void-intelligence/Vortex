// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class LReLU : Utility.BaseActivation
    {
        public double Alpha { get; set; }

        public LReLU(LReLUSettings settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => System.Math.Max(input, Alpha);

        protected override double Derivative(double input) => (input > 0) ? 1 : Alpha;

        public override Utility.EActivationType Type() => Utility.EActivationType.LReLU;
    }

    public sealed class LReLUSettings : Utility.ActivationSettings
    {
        public LReLUSettings(double alpha = 0.01)
        {
            Alpha = alpha;
        }

        public double Alpha { get; private set; }

        public override Utility.EActivationType Type() => Utility.EActivationType.LReLU;
    }
}
