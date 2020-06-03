﻿// Copyright © 2020 Void-Intelligence All Rights Reserved.

using Nomad.Core;
using Vortex.Activation.Utility;

namespace Vortex.Activation.Kernels
{
    public sealed class HardTanh : BaseActivation
    {
        public override Matrix Forward(Matrix input)
        {
            return input.Map(Activate);
        }

        public override Matrix Backward(Matrix input)
        {
            return input.Map(Derivative);
        }

        public override double Activate(double input)
        {
            return input < -1 ? -1 : input > 1 ? 1 : input;
        }

        public override double Derivative(double input)
        {
            return input < -1 ? 0 : input > 1 ? 0 : 1;
        }

        public override EActivationType Type()
        {
            return EActivationType.HardTanh;
        }
    }
}
