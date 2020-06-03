// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public sealed class Uniform : BaseInitializer
    {
        public Uniform(double scale = 1.0, double min = -0.5, double max = 0.5) : base(scale, min, max)
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
}