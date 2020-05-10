// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Regularization.Utility
{
    public abstract class BaseRegularization
    {
        public double Lambda { get; set; }

        protected BaseRegularization(double lambda)
        {
            Lambda = lambda;
        }

        public abstract double CalculateNorm(Matrix input);

        public abstract ERegularizationType Type();
    }
}
