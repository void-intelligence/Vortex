// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    /// <summary>
    /// Lasso-Ridge Regularization
    /// </summary>
    public sealed class L1L2 : BaseRegularization
    {
        public L1L2(double lambda = 1) : base(lambda)
        {
        }

        public override double CalculateNorm(Matrix input)
        {
            return (input.AbsoluteNorm() + input.EuclideanNorm()) * Lambda;
        }

        public override ERegularizationType Type()
        {
            return ERegularizationType.L1L2;
        }
    }
}
