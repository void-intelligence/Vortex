// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Swish : BaseActivation
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
            return Exp(-input) + 1.0;
        }

        public override double Derivative(double input)
        {
            return (1.0 + Exp(input) + input) / Pow(1.0 + Exp(input), 2.0);
        }

        public override EActivationType Type()
        {
            return EActivationType.Swish;
        }
    }
}