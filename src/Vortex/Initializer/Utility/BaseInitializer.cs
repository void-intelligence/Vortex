// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;

namespace Vortex.Initializer.Utility
{
    public abstract class BaseInitializer
    {
        public double Scale { get; set; }
        public double Min { get; set; }
        public double Max { get; set; }

        protected BaseInitializer(double scale = 1.0, double min = -0.5, double max = 0.5)
        {
            Min = min;
            Max = max;
            Scale = scale;
        }

        public abstract Matrix Initialize(Matrix w);
        public abstract EInitializerType Type();
    }
}
