// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Logit : BaseActivation
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
            return Log(input / (1 - input));
        }

        public override double Derivative(double input)
        {
            return -1 / Pow(input, 2) - 1 / Pow(1 - input, 2);
        }

        public override EActivationType Type()
        {
            return EActivationType.Logit;
        }
    }
}
