﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Initializer.Utility
{
    public interface IInitializer
    {
        public Matrix Initialize(Matrix w);
        public EInitializerType Type();
    }
}