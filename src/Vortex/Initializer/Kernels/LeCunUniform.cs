// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public class LeCunUniformKernel : BaseInitializerKernel
    {
        public LeCunUniformKernel(LeCunUniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InRandomize(Sqrt(3.0 / w.Columns));
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.LeCunUniform;
        }
    }

    public class LeCunUniform : BaseInitializer
    {
        public LeCunUniform(double min = -0.5, double max = 0.5, double scale = 0.01) : base(min, max, scale)
        {
        }

        public override EInitializerType Type()
        {
            return EInitializerType.LeCunUniform;
        }
    }
}