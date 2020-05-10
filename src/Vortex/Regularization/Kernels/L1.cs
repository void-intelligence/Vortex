// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    /// <summary>
    /// Lasso Regularization
    /// </summary>
    public sealed class L1 : BaseRegularization
    {
        public L1(double lambda = 1) : base(lambda)
        {
        }

        public override double CalculateNorm(Matrix input)
        {
            return input.AbsoluteNorm() * Lambda;
        }

        public override ERegularizationType Type()
        {
            return ERegularizationType.L1;
        }
    }

}
