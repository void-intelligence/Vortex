// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class SigmoidKernel : BaseActivationKernel
    {
        public SigmoidKernel(Sigmoid settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => 1.0 / (1 + Exp(-input));

        protected override double Derivative(double input) => (1.0 / (1 + Exp(-input))) * (1 - (1.0 / (1 + Exp(-input))));
        
        public override EActivationType Type() => EActivationType.Sigmoid;
    }

    public sealed class Sigmoid : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Sigmoid;
    }
}
