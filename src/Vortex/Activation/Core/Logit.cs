// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Activation
{
    public sealed class Logit : Utility.BaseActivation
    {
        public Logit(LogitSettings settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => System.Math.Log(input / (1 - input));
        
        protected override double Derivative(double input) => (-1 / System.Math.Pow(input, 2)) - (1 / (System.Math.Pow((1 - input), 2)));
        
        public override Utility.EActivationType Type() => Utility.EActivationType.Logit;
    }

    public sealed class LogitSettings : Utility.ActivationSettings
    {
        public override Utility.EActivationType Type() => Utility.EActivationType.Logit;
    }
}
