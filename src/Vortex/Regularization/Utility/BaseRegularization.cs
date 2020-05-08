// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Regularization.Utility
{
    public abstract class BaseRegularizationKernel
    {
        public double Lambda { get; set; }

        protected BaseRegularizationKernel(BaseRegularization settings) {}

        public abstract double CalculateNorm(Matrix input);

        public abstract ERegularizationType Type();
    }

    public abstract class BaseRegularization
    {
        public abstract ERegularizationType Type();
    }
}
