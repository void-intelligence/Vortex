﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class NadamKernel : BaseOptimizerKernel
    {
        public NadamKernel(Nadam settings) : base(settings)
        {
        }
        public NadamKernel(double alpha = 0.001) : base(new Nadam(alpha))
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
            return EOptimizerType.Nadam;
        }
    }

    public sealed class Nadam : BaseOptimizer
    {
        public override EOptimizerType Type()
        {
            return EOptimizerType.Nadam;
        }

        public Nadam(double alpha = 0.001) : base(alpha)
        {
        }
    }
}
