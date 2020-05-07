// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class ArctanKernel : BaseActivationKernel
    {
        public ArctanKernel(Arctan settings = null) : base(settings) { }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Atan(input);
        
        protected override double Derivative(double input) => 1 / (1 + Pow(input, 2));

        public override EActivationType Type() => EActivationType.Arctan;
    }

    public sealed class Arctan : BaseActivation
    {
        public override EActivationType Type() => EActivationType.Arctan;
    }
}
