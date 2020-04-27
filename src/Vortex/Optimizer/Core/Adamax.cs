// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Adamax : Utility.BaseOptimizer
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }

        public Adamax(AdamaxSettings settings) : base(settings)
        {
            BetaPrimary = settings.BetaPrimary;
            BetaSecondary = settings.BetaSecondary;
        }

        public override Matrix CalculateDeltaW(Matrix w, Matrix dJdW)
        {
            return null;
        }

        public override Matrix CalculateDeltaB(Matrix b, Matrix dJdB)
        {
            return null;
        }


        public override EOptimizerType Type() => EOptimizerType.Adamax;
    }

    public sealed class AdamaxSettings : OptimizerSettings
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Adamax;

        public AdamaxSettings(double betaPrimary, double betaSecondary, double alpha = 0.001) : base(alpha)
        {
            BetaPrimary = betaPrimary;
            BetaSecondary = betaSecondary;
        }
    }
}
