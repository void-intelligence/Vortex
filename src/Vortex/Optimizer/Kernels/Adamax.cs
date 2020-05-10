﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdamaxKernel : BaseOptimizerKernel
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }

        public AdamaxKernel(Adamax settings) : base(settings)
        {
            BetaPrimary = settings.BetaPrimary;
            BetaSecondary = settings.BetaSecondary;
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
            return EOptimizerType.Adamax;
        }
    }

    public sealed class Adamax : BaseOptimizer
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public override EOptimizerType Type()
        {
            return EOptimizerType.Adamax;
        }

        public Adamax(double betaPrimary, double betaSecondary, double alpha = 0.001) : base(alpha)
        {
            BetaPrimary = betaPrimary;
            BetaSecondary = betaSecondary;
        }
    }
}
