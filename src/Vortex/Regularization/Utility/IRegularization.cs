// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Regularization.Utility
{
    public interface IRegularization
    {
        public double CalculateNorm(Matrix input);

        public ERegularizationType Type();

    }
}