// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class LReluKernel : BaseActivationKernel
    {
        public double Alpha { get; set; }

        public LReluKernel(LRelu settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => Max(input, Alpha);

        protected override double Derivative(double input) => (input > 0) ? 1 : Alpha;

        public override EActivationType Type() => EActivationType.LRelu;
    }

    public sealed class LRelu : BaseActivation
    {
        public LRelu(double alpha = 0.01)
        {
            Alpha = alpha;
        }

        public double Alpha { get; private set; }

        public override EActivationType Type() => EActivationType.LRelu;
    }
}
