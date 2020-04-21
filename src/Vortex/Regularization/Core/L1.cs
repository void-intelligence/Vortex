// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization
{
    /// <summary>
    /// Lasso Regularization
    /// </summary>
    public sealed class L1 : Utility.BaseRegularization
    {
        public L1(L1Settings settings) : base(settings) { Lambda = settings.Lambda; }

        public override double CalculateNorm(Matrix input) => (input.AbsoluteNorm() * Lambda);

        public override string ToString() => Type().ToString();

        public override ERegularizationType Type() => ERegularizationType.L1;
    }

    public sealed class L1Settings : RegularizationSettings
    {
        public L1Settings(double lambda)
        {
            Lambda = lambda;
        }

        public override ERegularizationType Type() => ERegularizationType.L1;

        public double Lambda { get; set; }
    }

}
