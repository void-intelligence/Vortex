// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Exponential : BaseActivation
    {
        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Exp(input);

        protected override double Derivative(double input) => Exp(input);

        public override EActivationType Type() => EActivationType.Exponential;
    }
}