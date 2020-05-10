// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class RmsPropKernel : BaseOptimizerKernel
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }

        public RmsPropKernel(RmsProp settings) : base(settings)
        {
            Rho = settings.Rho;
            Epsilon = settings.Epsilon;
        }

        public RmsPropKernel(double alpha = 0.01, double rho = 0.9, double epsilon = 0.00001) : base(new RmsProp(alpha, rho, epsilon))
        {
            Rho = rho;
            Epsilon = epsilon;
        }

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            dJdX.Cache[0] = Rho * dJdX.Cache[0] + ((1.0 - Rho) * (dJdX.Hadamard(dJdX)));
            Matrix oneover = (dJdX.Cache[0].Map(Math.Sqrt) + dJdX.Cache[0].Fill(Epsilon)).OneOver();
            return x - Alpha * dJdX.Hadamard(oneover);

        }

        public override EOptimizerType Type() => EOptimizerType.RmsProp;
    }

    public sealed class RmsProp : Utility.BaseOptimizer
    {
        public double Rho { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.RmsProp;

        public RmsProp(double alpha = 0.01, double rho = 0.9, double epsilon = 0.00001) : base(alpha)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
    }
}
