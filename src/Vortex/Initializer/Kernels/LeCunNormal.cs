﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Matrix;
using Nomad.Utility;
using Vortex.Initializer.Utility;
using static System.Math;

namespace Vortex.Initializer.Kernels
{
    public class LeCunNormalKernel : BaseInitializerKernel
    {
        private int _h;
        private double Method(double input)
        {
            return input * Sqrt(1.0 / _h);
        }

        public LeCunNormalKernel(LeCunNormal initializer) : base(initializer)
        {
        }

        public override Matrix Initialize(Matrix w)
        {
            _h = w.Columns;
            Matrix mat = w.Duplicate();
            mat.InRandomize(Min, Max, EDistribution.Gaussian);
            mat.InMap(Method);
            mat *= Scale;
            return mat;
        }

        public override EInitializerType Type() => EInitializerType.LeCunNormal;
    }

    public class LeCunNormal : BaseInitializer
    {
        public LeCunNormal(double min = -0.5, double max = 0.5, double scale = 0.01) : base(min, max, scale)
        {
        }

        public override EInitializerType Type() => EInitializerType.LeCunNormal;
    }
}