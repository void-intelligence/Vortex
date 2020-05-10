// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public class NormalKernel : BaseInitializerKernel
    {
        public NormalKernel(Normal initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Gaussian);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Normal;
        }
    }

    public class Normal : BaseInitializer
    {
        public Normal(double min = -0.5, double max = 0.5, double scale = 0.01) : base(min, max, scale)
        {
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Normal;
        }
    }
}