// Copyright © 2020 Void-Intelligence All Rights Reserved.

using System;
using Nomad.Matrix;

namespace Vortex.Initializer.Utility
{
    public abstract class BaseInitializerKernel
    {
        public double Scale { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }

        protected BaseInitializerKernel(BaseInitializer init)
        {
            Min = init.Min;
            Max = init.Max;
            Scale = init.Scale;
        }

        public abstract Matrix Initialize(Matrix w);
        public abstract EInitializerType Type();
    }

    public abstract class BaseInitializer
    {
        public double Scale { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }

        protected BaseInitializer(double min = -0.5, double max = 0.5, double scale = 1.0)
        {
            Min = min;
            Max = max;
            Scale = scale;
        }
        public abstract EInitializerType Type();
    }
}
