// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Softplus : BaseActivation
    {
        public override Matrix Forward(Matrix input)
        {
            return input.Map(Activate);
        }

        public override Matrix Backward(Matrix input)
        {
            return input.Map(Derivative);
        }

        protected override double Activate(double input)
        {
            return Log(1 + Exp(input));
        }

        protected override double Derivative(double input)
        {
            return Exp(input) / (1 + Exp(input));
        }

        public override EActivationType Type()
        {
            return EActivationType.Softplus;
        }
    }
}
