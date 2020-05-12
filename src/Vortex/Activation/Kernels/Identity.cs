// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class Identity : BaseActivation
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
            return input;
        }

        public override double Derivative(double input)
        {
            return 1;
        }

        public override EActivationType Type()
        {
            return EActivationType.Identity;
        }
    }
}
