﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using System;
using Vortex.Decay.Utility;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class Nadam : BaseOptimizer
    {
        public double Beta1 { get; set; }
        public double Beta2 { get; set; }
        public double T { get; private set; }
        public double Epsilon { get; set; }

#nullable enable
        public Nadam(double alpha = 0.01, IDecay? decay = null, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 0.00001) : base(alpha, decay)
        {
            T = 0;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
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

                // SDW
                dJdX.Cache.Add(dJdX.Fill(0));
                // SDW Corrected
                dJdX.Cache.Add(dJdX.Fill(0));

                // Iteration T on dJdX
                dJdX.Cache.Add(Matrix.Zero(1));
            }

            // Iteration T
            T = dJdX.Cache[^1][0, 0]++;

            // Nesterov  Momentum (VDW)
            dJdX.Cache[0] =  dJdX.Cache[0] * Beta1 + dJdX * (1.0 - Beta1);

            // Nesterov Momentum Corrected (VDW C)
            var tt1 = Math.Pow(1.0 - Beta1, T);
            dJdX.Cache[1] = dJdX.Cache[0].HadamardDivision(dJdX.Fill(tt1)) +  dJdX.HadamardDivision(dJdX.Fill(tt1)) * (1.0 - Beta1);
            

            // RMS-Prop (SDW)
            dJdX.Cache[2] = dJdX.Cache[2] * Beta2 + dJdX.Hadamard(dJdX).Hadamard(dJdX.Hadamard(dJdX)) * (1.0 - Beta2);

            // RMS-Prop Corrected (SDW C)
            var tt2 = Math.Pow(1.0 - Beta2, T);
            dJdX.Cache[3] = dJdX.Cache[2].HadamardDivision(dJdX.Cache[2].Fill(tt2));

            // X optimization
            return x - dJdX.Cache[1].HadamardDivision(dJdX.Cache[3].Map(Math.Sqrt) + dJdX.Cache[3].Fill(Epsilon)) * Alpha;
        }

        public override EOptimizerType Type()
        {
            return EOptimizerType.Nadam;
        }
    }
}
