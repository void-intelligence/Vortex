// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;

namespace Vortex.Layer.Utility
{
    public interface ILayer
    {
        public Matrix Forward(Matrix inputs);
        public Matrix Backward(Matrix dA);
        public ELayerType Type();
    }
}