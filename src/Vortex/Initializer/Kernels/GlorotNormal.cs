// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public sealed class GlorotNormal : BaseInitializer
    {
        private int _denom;
        private double Method(double input)
        {
            return input * Sqrt(2.0 / _denom);
        }

        public GlorotNormal(double scale = 1.0, double min = -0.5, double max = 0.5) : base(scale, min, max)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            _denom = w.Columns + w.Rows;

            var mat = w.Duplicate();
            mat.InRandomize(Min,Max,EDistribution.Gaussian);
            mat.InMap(Method);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.GlorotNormal;
        }
    }
}