﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Normalization.Utility
{
    public abstract class BaseNormalization : INormalization
    {
        public abstract Matrix Normalize(Matrix input);
        public abstract ENormalizationType Type();
    }
}
