// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Cost.Utility
{
    public interface ICost
    {
        public double Forward(Matrix actual, Matrix expected);

        public Matrix Backward(Matrix actual, Matrix expected);

        public ECostType Type();
    }
}