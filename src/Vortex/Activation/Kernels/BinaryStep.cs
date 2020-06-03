// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class BinaryStep : BaseActivation
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
            return input < 0 ? 0 : 1;
        }

        public override double Derivative(double input)
        {
            return 0;
        }

        public override EActivationType Type()
        {
            return EActivationType.BinaryStep;
        }
    }
}