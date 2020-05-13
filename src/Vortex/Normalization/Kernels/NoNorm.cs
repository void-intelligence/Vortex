// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Normalization.Utility;

namespace Vortex.Normalization.Kernels
{
    public class NoNorm : BaseNormalization
    {
        public override Matrix Normalize(Matrix input)
        {
            return input;
        }

        public override ENormalizationType Type()
        {
            return ENormalizationType.NoNorm;
        }
    }
}
