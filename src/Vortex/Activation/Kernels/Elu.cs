// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class Elu: BaseActivation
    {
        public double Alpha { get; set; }

        public Elu(double alpha)
        {
            Alpha = alpha;
        }

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
            return input >= 0 ? input : Alpha * (Exp(input) - 1);
        }

        protected override double Derivative(double input)
        {
            return input >= 0 ? 1 : Alpha * Exp(input);
        }

        public override EActivationType Type()
        {
            return EActivationType.Elu;
        }
    }
}
