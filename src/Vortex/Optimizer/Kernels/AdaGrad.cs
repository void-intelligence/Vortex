// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaGradKernel : BaseOptimizerKernel
    {
        public double Epsilon { get; set; }

        public AdaGradKernel(AdaGrad settings): base(settings)
        {
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            return null;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.AdaGrad;
    }

    public sealed class AdaGrad : Utility.Optimizer
    {
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.AdaGrad;

        public AdaGrad(double epsilon, double alpha = 0.001) : base(alpha)
        {
            Epsilon = epsilon;
        }
    }
}
