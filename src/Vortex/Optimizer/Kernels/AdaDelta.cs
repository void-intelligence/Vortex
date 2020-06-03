// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaDelta : BaseOptimizer
    {
        public  double Rho { get; set; }
        public double Epsilon { get; set; }

#nullable enable
        public AdaDelta(double alpha = 0.01, IDecay? decay = null, double rho = 0.95, double epsilon = 0.00001) : base(alpha, decay)
        {
            Rho = rho;
            Epsilon = epsilon;
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                // Cache1
                dJdX.Cache.Add(dJdX.Fill(0));
                // Cache2
                dJdX.Cache.Add(dJdX.Fill(0));

                // Delta
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            dJdX.Cache[^1][0, 0]++;

            // Cache 1
            dJdX.Cache[0] = dJdX.Cache[0] * Rho + dJdX.Hadamard(dJdX) * (1.0 - Rho);
            
            // Delta
            dJdX.Cache[2] = (dJdX.Cache[1] + dJdX.Cache[1].Fill(Epsilon)).Map(Math.Sqrt).Hadamard(dJdX.HadamardDivision(dJdX.Cache[0] + dJdX.Cache[0].Fill(Epsilon)));

            // Cache 2
            dJdX.Cache[1] = dJdX.Cache[2]  * Rho + dJdX.Cache[2].Hadamard(dJdX.Cache[2]) * (1.0 - Rho);

            return x - dJdX.Cache[2] * Alpha;
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.AdaGrad;
        }
    }
}