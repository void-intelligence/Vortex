// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    /// <summary>
    /// Lasso Regularization
    /// </summary>
    public sealed class L1Kernel : BaseRegularizationKernel
    {
        public L1Kernel(L1 settings) : base(settings) { Lambda = settings.Lambda; }

        public override double CalculateNorm(Matrix input) => (input.AbsoluteNorm() * Lambda);

        public override string ToString() => Type().ToString();

        public override ERegularizationType Type() => ERegularizationType.L1;
    }

    public sealed class L1 : Utility.Regularization
    {
        public L1(double lambda)
        {
            Lambda = lambda;
        }

        public override ERegularizationType Type() => ERegularizationType.L1;

        public double Lambda { get; set; }
    }

}
