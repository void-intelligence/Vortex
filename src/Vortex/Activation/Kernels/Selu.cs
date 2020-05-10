// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Selu : BaseActivation
    {
        public const double Alpha = 1.6732632423543772848170429916717;

        public const double Lambda = 1.0507009873554804934193349852946;

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
            return input > 0 ? input : Alpha * (Exp(input) - 1);
        }

        protected override double Derivative(double input)
        {
            return input > 0 ? Lambda : Lambda * Alpha * Exp(input);
        }

        public override EActivationType Type()
        {
            return EActivationType.Selu;
        }
    }
}
