// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Vortex.Initializer.Utility;

namespace Vortex.Initializer.Kernels
{
    public class ZeroKernel : BaseInitializerKernel
    {
        public ZeroKernel(Zero initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            var mat = w.Duplicate();
            mat.InFill(0);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type()
        {
            return EInitializerType.Zero;
        }
    }

    public class Zero : BaseInitializer
    {
        public Zero(double min = -0.5, double max = 0.5, double scale = 1.0) : base(min, max, scale)
        {
        }
    
        public override EInitializerType Type()
        {
            return EInitializerType.Zero;
        }
    }
}