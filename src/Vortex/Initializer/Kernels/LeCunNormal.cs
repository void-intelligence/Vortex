// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public sealed class LeCunNormal : BaseInitializer
    {
        private int _h;
        private double Method(double input)
        {
            return input * Sqrt(1.0 / _h);
        }

        public LeCunNormal(double scale = 1.0, double min = -0.5, double max = 0.5) : base(scale, min, max)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            _h = w.Columns;
            var mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Gaussian);
            mat.InMap(Method);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.LeCunNormal;
        }
    }
}