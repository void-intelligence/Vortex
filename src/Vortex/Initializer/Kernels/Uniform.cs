// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public class UniformKernel : BaseInitializerKernel
    {
        public UniformKernel(Uniform initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InRandomize(Min, Max);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Uniform;
        }
    }

    public class Uniform : BaseInitializer
    {
        public Uniform(double min = -0.5, double max = 0.5, double scale = 0.01) : base(min, max, scale)
        {
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Uniform;
        }
    }
}