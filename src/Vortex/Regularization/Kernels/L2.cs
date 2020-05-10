// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    /// <summary>
    /// Ridge Regularization
    /// </summary>
    public sealed class L2 : BaseRegularization
    {
        public L2(double lambda = 1) : base(lambda)
        {
        }

        public override double CalculateNorm(Matrix input)
        {
            return input.EuclideanNorm() * Lambda;
        }

        public override ERegularizationType Type()
        {
            return ERegularizationType.L2;
        }
    }
}
