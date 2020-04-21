// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Sigmoid : Utility.BaseActivation
    {
        public Sigmoid(SigmoidSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => 1.0 / (1 + System.Math.Exp(-input));

        protected override double Derivative(double input) => (1.0 / (1 + System.Math.Exp(-input))) * (1 - (1.0 / (1 + System.Math.Exp(-input))));
        
        public override Utility.EActivationType Type() => Utility.EActivationType.Sigmoid;

        public override string ToString() => Type().ToString();
    }

    public sealed class SigmoidSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Sigmoid;
    }
}
