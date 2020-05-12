// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Tanh : BaseActivation
    {
        public override Matrix Forward(Matrix input)
        {
            return input.Map(Activate);
        }

        public override Matrix Backward(Matrix input)
        {
            return input.Map(Derivative);
        }

        public override double Activate(double input)
        {
            return (Exp(input) - Exp(-input)) / (Exp(input) + Exp(-input));
        }

        public override double Derivative(double input)
        {
            return 1 - Pow((Exp(input) - Exp(-input)) / (Exp(input) + Exp(-input)), 2);
        }

        public override EActivationType Type()
        {
            return EActivationType.Tanh;
        }
    }
}
