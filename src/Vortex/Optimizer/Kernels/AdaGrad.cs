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

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));

            // Iteration T
            dJdX.Cache[^1][0, 0]++;
            return null;
        }
        
        public override EOptimizerType Type()
        {
            return EOptimizerType.AdaGrad;
        }
    }

    public sealed class AdaGrad : BaseOptimizer
    {
        public double Epsilon { get; set; }
        public override EOptimizerType Type()
        {
            return EOptimizerType.AdaGrad;
        }

        public AdaGrad(double epsilon, double alpha = 0.001) : base(alpha)
        {
            Epsilon = epsilon;
        }
    }
}
