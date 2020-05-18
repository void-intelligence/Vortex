// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Pooler.Utility
{
    public interface IPooler
    {
        public Matrix Pad(Matrix x, Matrix filter);

        public Matrix Pool(Matrix x, Matrix filter);

        public EPoolerType Type();
    }
}