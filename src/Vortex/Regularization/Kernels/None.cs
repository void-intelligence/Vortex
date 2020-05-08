// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Regularization.Utility;

namespace Vortex.Regularization.Kernels
{
    public sealed class NoneKernel : BaseRegularizationKernel
    {
        public NoneKernel(None settings = null) : base(settings) { }

        public override double CalculateNorm(Matrix input) => 0;

        public override string ToString() => Type().ToString();

        public override ERegularizationType Type() => ERegularizationType.None;
    }

    public sealed class None : BaseRegularization
    {
        public override ERegularizationType Type() => ERegularizationType.None;
    }
}
