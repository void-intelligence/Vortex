// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Regularization.Utility
{
    public abstract class Regularization
    {
        public double Lambda { get; set; }

        public Regularization(RegularizationSettings settings) { }

        public abstract double CalculateNorm(Matrix input);

        public abstract ERegularizationType Type();
    }

    public abstract class RegularizationSettings
    {
        public abstract ERegularizationType Type();
    }
}
