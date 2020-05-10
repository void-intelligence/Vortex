// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NesterovMomentumKernel : BaseOptimizerKernel
    {
      

        public NesterovMomentumKernel(NesterovMomentum settings) : base(settings)
        {
        }

        public NesterovMomentumKernel(double alpha = 0.001) : base(new NesterovMomentum(alpha))
        {
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
            return EOptimizerType.NesterovMomentum;
        }
    }

    public sealed class NesterovMomentum : BaseOptimizer
    {
        public override EOptimizerType Type()
        {
            return EOptimizerType.NesterovMomentum;
        }

        public NesterovMomentum(double alpha = 0.001) : base(alpha)
        {
        }
    }
}


