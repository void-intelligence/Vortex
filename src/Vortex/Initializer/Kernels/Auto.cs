// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public class AutoKernel : BaseInitializerKernel
    {
        public AutoKernel(Auto initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            Matrix mat = w.Duplicate();
            mat.InRandomize(-0.5, 0.5, EDistribution.Gaussian);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.Auto;
    }

    public class Auto : BaseInitializer
    {
        public Auto(double scale = 1.0) : base(-0.5, 0.5, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.Auto;
    }
}