// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization
{
    /// <summary>
    /// Ridge Regularization
    /// </summary>
    public sealed class L2 : Utility.BaseRegularization
    {
        public L2(L2Settings settings) : base(settings) { Lambda = settings.Lambda; }

        public override double CalculateNorm(Matrix input) => (input.EuclideanNorm() * Lambda);

        public override string ToString() => Type().ToString();

        public override ERegularizationType Type() => ERegularizationType.L2;
    }

    public sealed class L2Settings : RegularizationSettings
    {
        public L2Settings(double lambda)
        {
            Lambda = lambda;
        }

        public override ERegularizationType Type() => ERegularizationType.L2;

        public double Lambda { get; set; }
    }
}
