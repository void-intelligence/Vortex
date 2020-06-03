// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Relu : BaseActivation
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
            return Max(input, 0);
        }

        public override double Derivative(double input)
        {
            return input > 0 ? 1 : 0;
        }

        public override EActivationType Type()
        {
            return EActivationType.Relu;
        }
    }
}
