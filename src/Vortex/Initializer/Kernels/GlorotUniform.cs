// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public sealed class GlorotUniform : BaseInitializer
    {
        public GlorotUniform(double scale = 1.0, double min = -0.5, double max = 0.5) : base(scale, min, max)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InRandomize(Sqrt(6.0 / (w.Columns + w.Rows)));
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.GlorotUniform;
        }
    }
}