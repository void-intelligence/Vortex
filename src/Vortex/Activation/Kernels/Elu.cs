// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Activation.Utility;
using static System.Math;

namespace Vortex.Activation.Kernels
{
    public sealed class EluKernel : BaseActivationKernel
    {
        public double Alpha { get; set; }

        public EluKernel(Elu settings) : base(settings)
        {
            Alpha = settings.Alpha;
        }

        public override Matrix Forward(Matrix input) => input.Map(Activate);

        public override Matrix Backward(Matrix input) => input.Map(Derivative);

        protected override double Activate(double input) => (input >= 0) ? input : Alpha * (Exp(input) - 1);

        protected override double Derivative(double input) => (input >= 0) ? 1 : Alpha * Exp(input);

        public override EActivationType Type() => EActivationType.Elu;
    }

    public sealed class Elu : BaseActivation
    {
        public Elu(double alpha = 0.01)
        {
            Alpha = alpha;
        }

        public double Alpha { get; }

        public override EActivationType Type() => EActivationType.Elu;
    }
}
