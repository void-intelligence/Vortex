// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaGrad : BaseOptimizer
    {
        public double Epsilon { get; set; }

#nullable enable
        public AdaGrad(double alpha = 0.01, IDecay? decay = null, double epsilon = 0.00001) : base(alpha, decay)
        {
            Epsilon = epsilon;
        }
#nullable disable

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

            dJdX.Cache[0] = dJdX.Hadamard(dJdX);
            return x - dJdX.HadamardDivision(dJdX.Cache[0].Map(Math.Sqrt) + dJdX.Cache[0].Fill(Epsilon)) * Alpha;
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.AdaGrad;
        }
    }
}
