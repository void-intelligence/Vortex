// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Core;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdaMax : BaseOptimizer
    {
        public double Beta1 { get; set; }
        public double Beta2 { get; set; }
        public double Epsilon { get;set; }
        public double T { get; private set; }

#nullable enable
        public AdaMax(double alpha = 0.01, IDecay? decay = null, double b1 = 0.9, double b2 = 0.999, double epsilon = 0.00001) : base(alpha, decay)
        {
            T = 0;
            Epsilon = epsilon;
            Beta1 = b1;
            Beta2 = b2;
        }
#nullable disable

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            if (dJdX.Cache.Count == 0)
            {
                // VDW
                dJdX.Cache.Add(dJdX.Fill(0));
                // VDW Corrected
                dJdX.Cache.Add(dJdX.Fill(0));

                // MaxDw
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            T = dJdX.Cache[^1][0, 0]++;

            // Momentum (VDW)
            dJdX.Cache[0] = dJdX.Cache[0] * Beta1 + dJdX * (1.0 - Beta1);

            // Momentum Corrected (VDW C)
            var tt1 = Math.Pow(1.0 - Beta1, T);
            dJdX.Cache[1] = dJdX.Cache[0].HadamardDivision(dJdX.Cache[0].Fill(tt1));

            // Max (MaxDw)
            for (var i = 0; i < dJdX.Cache[2].Rows; i++)
            for (var j = 0; j < dJdX.Cache[2].Columns; j++)
                dJdX.Cache[2][i, j] = Math.Max(Beta2 * dJdX.Cache[2][i, j], Math.Abs(dJdX[i, j]));
            
            // X optimization
            return x - dJdX.Cache[1].HadamardDivision(dJdX.Cache[2] + dJdX.Cache[2].Fill(Epsilon)) * Alpha;
        }


        public override EOptimizerType Type()
        {
            return EOptimizerType.Adamax;
        }
    }
}
