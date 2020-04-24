// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer
{
    public sealed class Adam : Utility.BaseOptimizer
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimary_T { get; set; }
        public double BetaSecondary_T { get; set; }
        public double Epsilon { get; set; }
        public Adam(AdamSettings settings) : base(settings)
        {
            BetaPrimary = settings.BetaPrimary;
            BetaSecondary = settings.BetaSecondary;
            BetaPrimary_T = settings.BetaPrimary_T;
            BetaSecondary_T = settings.BetaSecondary_T;
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDelta(Matrix X, Matrix dJdX)
        {
            return null;
        }

        public override EOptimizerType Type() => EOptimizerType.Adam;
    }

    public sealed class AdamSettings : OptimizerSettings
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimary_T { get; set; }
        public double BetaSecondary_T { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Adam;

        public AdamSettings(double alpha, double betaPrimary, double betaSecondary, double betaPrimary_T, double betaSecondary_T, double epsilon) : base(alpha)
        {
            BetaPrimary = betaPrimary;
            BetaSecondary = betaSecondary;
            BetaPrimary_T = betaPrimary_T;
            BetaSecondary_T = betaSecondary_T;
            Epsilon = epsilon;
        }
    }
}
