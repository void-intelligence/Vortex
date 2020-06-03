// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Softsign : BaseActivation
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
            return input / (1 + Abs(input));
        }

        public override double Derivative(double input)
        {
            return input / Pow(1 + Abs(input), 2);
        }

        public override EActivationType Type()
        {
            return EActivationType.Softsign;
        }
    }
}
