// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Mish : Utility.Activation
    {
        public Mish(MishSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => input * System.Math.Tanh(System.Math.Log(1 + System.Math.Exp(input)));

        protected override double Derivative(double input)
        {
            double epsilon = 2 * System.Math.Exp(input) + System.Math.Exp(2 * input) + 2;

            double omega = 4 * (input + 1) + 4 * System.Math.Exp(2 * input) + 
                System.Math.Exp(3 * input) + System.Math.Exp(input) * (4 * input + 6);

            return ((System.Math.Exp(input) * omega) / System.Math.Pow(epsilon, 2));
        }

        public override Utility.EActivationType Type() => Utility.EActivationType.Mish;
        
        public override string ToString() => Type().ToString();
    }

    public sealed class MishSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Mish;
    }
}
