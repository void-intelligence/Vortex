// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Regularization.Utility
{
    public abstract class BaseRegularization
    {
        public double Dropout { get; set; }

        public double Lambda { get; set; }

        public BaseRegularization(RegularizationSettings settings) 
        {
            Dropout = settings.Dropout;
        }

        public abstract double CalculateNorm(Matrix input);

        public abstract ERegularizationType Type();
    }

    public abstract class RegularizationSettings
    {
        public double Dropout { get; set; }

        public RegularizationSettings(double dropout)
        {
            Dropout = dropout;
        }

        public abstract ERegularizationType Type();
    }
}
