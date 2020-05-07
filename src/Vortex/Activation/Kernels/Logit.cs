// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class LogitKernel : BaseActivationKernel
    {
        public LogitKernel(Logit settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Log(input / (1 - input));
        
        protected override double Derivative(double input) => (-1 / Pow(input, 2)) - (1 / (Pow((1 - input), 2)));
        
        public override EActivationType Type() => EActivationType.Logit;
    }

    public sealed class Logit : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Logit;
    }
}
