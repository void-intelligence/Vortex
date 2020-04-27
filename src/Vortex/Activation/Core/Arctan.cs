// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Arctan : Utility.BaseActivation
    {
        public Arctan(ArctanSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => System.Math.Atan(input);
        
        protected override double Derivative(double input) => 1 / (1 + System.Math.Pow(input, 2));

        public override Utility.EActivationType Type() => Utility.EActivationType.Arctan;
    }

    public sealed class ArctanSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Arctan;
    }
}
