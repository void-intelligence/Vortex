// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Sigmoid : BaseActivation
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
            return 1.0 / (1 + Exp(-input));
        }

        public override double Derivative(double input)
        {
            return 1.0 / (1 + Exp(-input)) * (1 - 1.0 / (1 + Exp(-input)));
        }

        public override EActivationType Type()
        {
            return EActivationType.Sigmoid;
        }
    }
}
