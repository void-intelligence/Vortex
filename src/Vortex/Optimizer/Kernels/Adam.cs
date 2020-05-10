// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Optimizer.Utility;

namespace Vortex.Optimizer.Kernels
{
    public sealed class AdamKernel : BaseOptimizerKernel
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimaryT { get; set; }
        public double BetaSecondaryT { get; set; }
        public double Epsilon { get; set; }
        public AdamKernel(Adam settings) : base(settings)
        {
            BetaPrimary = settings.BetaPrimary;
            BetaSecondary = settings.BetaSecondary;
            BetaPrimaryT = settings.BetaPrimaryT;
            BetaSecondaryT = settings.BetaSecondaryT;
            Epsilon = settings.Epsilon;
        }

        public override Matrix CalculateDelta(Matrix x, Matrix dJdX)
        {
            return null;
        }
        public override EOptimizerType Type() => EOptimizerType.Adam;
    }

    public sealed class Adam : Utility.BaseOptimizer
    {
        public double BetaPrimary { get; set; }
        public double BetaSecondary { get; set; }
        public double BetaPrimaryT { get; set; }
        public double BetaSecondaryT { get; set; }
        public double Epsilon { get; set; }
        public override EOptimizerType Type() => EOptimizerType.Adam;

        public Adam(double betaPrimary, double betaSecondary, double betaPrimaryT, double betaSecondaryT, double epsilon, double alpha = 0.001) : base(alpha)
        {
            BetaPrimary = betaPrimary;
            BetaSecondary = betaSecondary;
            BetaPrimaryT = betaPrimaryT;
            BetaSecondaryT = betaSecondaryT;
            Epsilon = epsilon;
        }
    }
}
