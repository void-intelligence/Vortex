// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class LRelu : BaseActivation
    {
        public double Alpha { get; set; }

        public LRelu(double alpha)
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

        public override double Activate(double input)
        {
            return Max(input, Alpha);
        }

        public override double Derivative(double input)
        {
            return input > 0 ? 1 : Alpha;
        }

        public override EActivationType Type()
        {
            return EActivationType.LRelu;
        }
    }
}
