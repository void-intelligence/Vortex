// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Normalization.Utility
{
    public interface INormalization
    {
        public Matrix Normalize(Matrix input);
        public ENormalizationType Type();
    }
}
