// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Arctan : BaseActivation
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
            return Atan(input);
        }

        protected override double Derivative(double input)
        {
            return 1 / (1 + Pow(input, 2));
        }

        public override EActivationType Type()
        {
            return EActivationType.Arctan;
        }
    }
}
