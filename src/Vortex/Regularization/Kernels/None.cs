// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    public sealed class None : BaseRegularization
    {
        public None() : base(0)
        {
        }

        public override double CalculateNorm(Matrix input)
        {
            return 0;
        }

        public override ERegularizationType Type()
        {
            return ERegularizationType.None;
        }
    }
}
