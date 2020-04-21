// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class BinaryStep : Utility.BaseActivation
    {
        public BinaryStep(BinaryStepSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input < 0) ? 0 : 1;

        protected override double Derivative(double input) => 0;

        public override Utility.EActivationType Type() => Utility.EActivationType.BinaryStep;

        public override string ToString() => Type().ToString();
    }

    public sealed class BinaryStepSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.BinaryStep;
    }
}
